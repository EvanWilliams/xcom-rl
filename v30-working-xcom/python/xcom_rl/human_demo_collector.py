"""
Record your own XCOM gameplay as (obs, action) pairs for BC pretraining.

Your mouse clicks on the OpenXcom window are captured via polling (no hook
installed), translated to game coordinates (320x200 internal resolution),
matched to the nearest entry in the ACTIONS table, and saved as JSONL in the
same format bc_pretrain.py reads.

HOW IT WORKS
  - The game starts fresh via bridge new_game command.
  - Real OS mouse events reach the game directly — the engine input filter is
    headless-only, so visible-mode clicks and cursor movement are native.
  - A background thread polls GetAsyncKeyState ~60 fps to detect which click
    happened, without installing any mouse hook.
  - env.step(0) is used after each recorded click to observe the updated state;
    it sends nothing to the bridge, so there is no double-click.

CONTROLS
  Mouse clicks   play normally in the OpenXcom window
  F12            stop recording and save
  Ctrl+C         stop recording and save

BATTLESCAPE
  Tactical combat is fully recordable.  While the engine is in the battlescape
  every left-click is captured as a grid cell (the only click form the env
  replays there — named battle buttons are suppressed), so all of these teach
  the policy correctly:
    - Firing a weapon in any mode: click the weapon in a soldier's hand → click
      Snap / Aimed / Auto in the action-menu popup → click the target tile.
      Works for every weapon that offers each mode.
    - Throwing / priming grenades: click the grenade in hand → click Throw (or
      Prime Grenade, then a fuse) → click the target tile.
    - Inventory: click the inventory icon → left-click an item to grab it →
      left-click a slot (or the ground) to drop it.  OpenXcom inventory is
      click-grab-click-drop, not drag, so both halves are captured.
  Right-clicks in the battlescape are NOT recorded — the action space has no
  right-click grid cell, so use left-click flows for anything you want in the
  demo (the game still responds to right-clicks natively; they just aren't
  saved).

MULTI-SESSION WORKFLOW
  You can record demos across multiple sessions and train on all of them.

  Option A — append all sessions into one file:
      py -3 -m xcom_rl.human_demo_collector                      # session 1 (creates file)
      py -3 -m xcom_rl.human_demo_collector --append             # session 2+ (appends)
      py -3 -m xcom_rl.bc_pretrain --demos python/data/human_demos.jsonl

  Option B — keep sessions as separate files, train on all at once:
      py -3 -m xcom_rl.human_demo_collector --out python/data/demos_session1.jsonl
      py -3 -m xcom_rl.human_demo_collector --out python/data/demos_session2.jsonl
      py -3 -m xcom_rl.bc_pretrain --demos python/data/demos_session*.jsonl

Usage:
    py -3 -m xcom_rl.human_demo_collector
    py -3 -m xcom_rl.human_demo_collector --out data/human_demos.jsonl
    py -3 -m xcom_rl.human_demo_collector --episodes 3
    py -3 -m xcom_rl.human_demo_collector --append   (add to existing file)
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import ACTIONS, ACTION_NAMES, XcomInterceptionEnv

_ROOT    = Path(__file__).parent.parent.parent
_OUT_DIR = _ROOT / "python" / "data"

_GAME_W          = 320      # OpenXcom internal resolution
_GAME_H          = 200
_WINDOW_SCALE    = 1        # visible-window magnification (320×200 at 1x)
_MATCH_THRESHOLD = 25       # max game-pixel distance for a named action match

_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_VK_F12     = 0x7B

# -- window helpers ------------------------------------------------------------

_u32 = ctypes.windll.user32
_cached_hwnd: int = 0
_target_pid:  int = 0   # PID of the engine WE launched; set in collect()


def _set_target_pid(pid: int) -> None:
    """Bind window lookup to a specific engine process and drop any cached hwnd."""
    global _target_pid, _cached_hwnd
    _target_pid  = pid
    _cached_hwnd = 0


def _hwnd_for_pid(pid: int) -> int:
    """Return the first visible top-level HWND owned by *pid*, or 0."""
    found: list = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def _cb(hwnd: int, _: int) -> bool:
        if _u32.IsWindowVisible(hwnd):
            wpid = ctypes.c_ulong()
            _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value == pid:
                found.append(hwnd)
                return False
        return True

    _u32.EnumWindows(_cb, 0)
    return found[0] if found else 0


def _find_openxcom_hwnd() -> int:
    """
    Return the hwnd of the OpenXcom window THIS collector launched.

    Matched by the engine's own child PID (bound via _set_target_pid) so we never
    latch onto a *different* OpenXcom window — e.g. a concurrent training run,
    whose engine windows are titled "OpenXcom #N  reward=...".  Only when the PID
    is not yet known (engine still starting) do we fall back to a title match.
    """
    global _cached_hwnd
    if _cached_hwnd and _u32.IsWindow(_cached_hwnd):
        return _cached_hwnd

    if _target_pid:
        _cached_hwnd = _hwnd_for_pid(_target_pid)
        return _cached_hwnd

    # Fallback: engine PID not known yet — best-effort title match.  Unreliable
    # if other OpenXcom windows are open, hence the PID path above is preferred.
    found: list = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def _enum_cb(hwnd: int, _: int) -> bool:
        buf = ctypes.create_unicode_buffer(256)
        _u32.GetWindowTextW(hwnd, buf, 256)
        if (buf.value.startswith("OpenXcom") or buf.value == "XCOM Human Demo") \
                and _u32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    _u32.EnumWindows(_enum_cb, 0)
    if found:
        _cached_hwnd = found[0]
    return _cached_hwnd


def _game_bounds() -> Optional[Tuple[int, int, int, int]]:
    """Return (left, top, width, height) of the OpenXcom client area in screen coords."""
    hwnd = _find_openxcom_hwnd()
    if not hwnd:
        return None
    pt   = ctypes.wintypes.POINT(0, 0)
    rect = ctypes.wintypes.RECT()
    _u32.ClientToScreen(hwnd, ctypes.byref(pt))
    _u32.GetClientRect(hwnd, ctypes.byref(rect))
    if rect.right <= 0 or rect.bottom <= 0:
        return None
    return pt.x, pt.y, rect.right, rect.bottom


def _in_window(sx: int, sy: int, b: Tuple[int, int, int, int]) -> bool:
    l, t, w, h = b
    return l <= sx < l + w and t <= sy < t + h


def _to_game(sx: int, sy: int, b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    l, t, w, h = b
    return int((sx - l) * _GAME_W / w), int((sy - t) * _GAME_H / h)


# -- action matching -----------------------------------------------------------

# Single pool of all actions with valid coordinates.  random_click /
# random_right_click have x=-1 and are excluded by the x >= 0 guard.
# Named buttons win when clicked exactly on their centre (0 px distance);
# grid cells (spaced every 3 px, always ≤2.1 px away) win everywhere else.
#
# The ``battle_*`` named buttons (battle_left_hand, battle_right_hand,
# battle_action_menu_*, battle_end_turn, battle_visible*, …) are EXCLUDED here.
# env._execute_action suppresses every non-"grid_" named click while
# env._in_battlescape is True (returns a no-op), so a recorded named battle
# action replays as a dead action and teaches BC nothing.  The agent reaches all
# of these — the hand weapon that OPENS the fire-mode menu, and the Snap / Aimed /
# Auto rows themselves — only via grid cells.  Because a named button at the exact
# click point wins at 0 px (beating every ≤2.1 px grid cell), leaving these in the
# pool meant a click dead-centre on the weapon slot or a fire-mode row recorded
# the suppressed battle_right_hand / battle_action_menu_* action instead of the
# replayable grid cell.  Dropping them guarantees such a click falls through to
# the nearest grid cell even if the battlescape-detection flag momentarily lags
# (see _poll_clicks and the idle broadcast-drain in collect()).  battle_pickup has
# x=None and is already excluded by the x>=0 guard (the env handles it explicitly,
# so it is not suppressed and would be safe to keep anyway).
_CLICKABLE = [
    (i, x, y, btn)
    for i, (name, x, y, btn) in enumerate(ACTIONS)
    if x is not None and x >= 0 and y is not None and y >= 0
    and not name.startswith("battle_")
]

# Grid-only pool (indices ≥260, all left-click).  Used while the engine is in
# the battlescape: env._execute_action SUPPRESSES every *named* (non-"grid_")
# click during battle (see env.py "if self._in_battlescape and not
# name.startswith('grid_')"), so the agent can only ever reach battle buttons —
# hand weapons, the fire-mode/Throw action menu, Prime Grenade, inventory
# grab/drop — via grid cells.  If we recorded the NAMED battle action a human
# happened to click dead-centre (e.g. battle_left_hand, battle_action_menu_1,
# battle_inventory), BC would train the policy to emit an action the env turns
# into a no-op.  Restricting battle matches to grid cells keeps every recorded
# battlescape click faithfully replayable.
_GRID_CLICKABLE = [
    (i, x, y, btn)
    for i, (name, x, y, btn) in enumerate(ACTIONS)
    if name.startswith("grid_")
]


def _match_action(gx: int, gy: int, btn: int, grid_only: bool = False) -> Optional[int]:
    # In the battlescape only grid cells are replayable (see _GRID_CLICKABLE).
    # Grid cells are all left-click, so a right-click in battle matches nothing
    # and is dropped — the action space has no right-click grid cell to hold it.
    pool = _GRID_CLICKABLE if grid_only else _CLICKABLE
    best_i, best_d = -1, float("inf")
    for i, x, y, b in pool:
        if b != btn:
            continue
        d = ((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    if best_i < 0 or best_d > _MATCH_THRESHOLD:
        return None
    return best_i


# -- obs serialisation ---------------------------------------------------------

def _obs_to_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.tolist()  if isinstance(v, np.ndarray)
           else v.item() if isinstance(v, np.generic)
           else int(v)
        for k, v in obs.items()
    }


# -- click poller --------------------------------------------------------------

def _poll_clicks(
    action_q: "queue.Queue[int]",
    stop_event: threading.Event,
    in_battle: Callable[[], bool],
    poll_interval: float = 0.016,   # ~60 fps
) -> None:
    """
    Poll mouse state at ~60 fps and enqueue matched actions for clicks inside
    the OpenXcom window.  No hook is installed; real OS events reach the game
    normally because the engine input filter only runs in headless mode.

    ``in_battle()`` reports whether the engine is currently in the battlescape;
    when it is, clicks are matched to grid cells only so every recorded action
    is one the env will actually replay (named battle clicks are suppressed —
    see _GRID_CLICKABLE).  This is what makes recording weapon fire modes
    (snap/aimed/auto), grenade throwing/priming, and inventory grab/drop
    produce faithful demos.
    """
    lb_prev = rb_prev = False

    while not stop_event.is_set():
        lb  = bool(_u32.GetAsyncKeyState(_VK_LBUTTON) & 0x8001)  # held OR pressed since last poll
        rb  = bool(_u32.GetAsyncKeyState(_VK_RBUTTON) & 0x8001)
        f12 = bool(_u32.GetAsyncKeyState(_VK_F12)     & 0x8000)

        if f12:
            print("\n[human_demo] F12 -- stopping recording ...", flush=True)
            stop_event.set()
            break

        pt = ctypes.wintypes.POINT()
        _u32.GetCursorPos(ctypes.byref(pt))
        b = _game_bounds()

        if b and _in_window(pt.x, pt.y, b):
            gx, gy = _to_game(pt.x, pt.y, b)
            grid_only = bool(in_battle())
            for pressed, prev, btn_int in ((lb, lb_prev, 1), (rb, rb_prev, 3)):
                if pressed and not prev:
                    matched = _match_action(gx, gy, btn_int, grid_only=grid_only)
                    if matched is not None:
                        action_q.put((matched, gx, gy))

        lb_prev, rb_prev = lb, rb
        time.sleep(poll_interval)


# -- window focus helpers ------------------------------------------------------

_HWND_TOPMOST    = ctypes.wintypes.HWND(-1)
_HWND_NOTOPMOST  = ctypes.wintypes.HWND(-2)
_SWP_NOMOVE      = 0x0002
_SWP_NOSIZE      = 0x0001
_SWP_SHOWWINDOW  = 0x0040


def _raise_game_window(hwnd: int) -> None:
    """
    Bring the game window to the front and centre it on the primary screen.

    SetForegroundWindow fails when the terminal owns the foreground lock, so we
    raise via SetWindowPos(HWND_TOPMOST) (which bypasses that restriction) and
    immediately drop back to HWND_NOTOPMOST.  This pops the window above the
    terminal once without pinning it always-on-top, so the user can still drag
    it anywhere afterwards.  The window is a fixed small 320×200 (launched with
    "-allowResize false") so it cannot be resized/maximised mid-session.  While
    raising we also move it to the exact centre of the screen so it always opens
    dead-centre regardless of any saved position.
    """
    _u32.ShowWindow(hwnd, 9)   # SW_RESTORE -- un-minimise if needed

    # Compute the centred top-left from the full window rect and screen size.
    wr = ctypes.wintypes.RECT()
    _u32.GetWindowRect(hwnd, ctypes.byref(wr))
    win_w = wr.right  - wr.left
    win_h = wr.bottom - wr.top
    scr_w = _u32.GetSystemMetrics(0)   # SM_CXSCREEN
    scr_h = _u32.GetSystemMetrics(1)   # SM_CYSCREEN
    x = max(0, (scr_w - win_w) // 2)
    y = max(0, (scr_h - win_h) // 2)

    _u32.SetWindowPos(
        hwnd, _HWND_TOPMOST, x, y, 0, 0,
        _SWP_NOSIZE | _SWP_SHOWWINDOW,
    )
    _u32.SetWindowPos(
        hwnd, _HWND_NOTOPMOST, x, y, 0, 0,
        _SWP_NOSIZE | _SWP_SHOWWINDOW,
    )
    # Rename so the window tiler (which matches "OpenXcom" by title) ignores it.
    _u32.SetWindowTextW(hwnd, "XCOM Human Demo")

    # Move cursor to window centre so the user can start clicking immediately.
    pt   = ctypes.wintypes.POINT(0, 0)
    rect = ctypes.wintypes.RECT()
    _u32.ClientToScreen(hwnd, ctypes.byref(pt))
    _u32.GetClientRect(hwnd, ctypes.byref(rect))
    cx = pt.x + rect.right  // 2
    cy = pt.y + rect.bottom // 2
    _u32.SetCursorPos(cx, cy)


# -- main recorder -------------------------------------------------------------

def collect(out_path: Path, n_episodes: int = 0, append: bool = False) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Rank 900: distinct from training (0..N) and eval (100+) so this env's
    # per-rank log/pid/tmp files never collide with a concurrent training run.
    # "openxcom_bridge_human.sock" parsed to rank 0, clashing with training env 0
    # and crashing on its locked live_env_0.json.tmp.
    env = XcomInterceptionEnv(
        headless     = False,
        socket_name  = "openxcom_bridge_900.sock",
        max_steps    = 999_999,
        # Small FIXED window at native 320×200 (_WINDOW_SCALE = 1). env passes
        # "-allowResize false" so the window can't grow/maximise mid-session (which
        # caused the inventory screen to drop/shift and blocked clicks); the title
        # bar still lets you drag it. _game_bounds() reads the live client rect
        # every poll, so clicks stay correctly mapped if you move the window.
        display_size = (_GAME_W * _WINDOW_SCALE, _GAME_H * _WINDOW_SCALE),
    )

    print("[human_demo] Starting OpenXcom ...", flush=True)
    obs, _ = env.reset()

    # Bind window lookup to the engine WE just launched, by its child PID.  This
    # is essential when a training run is active: those windows are also titled
    # "OpenXcom ...", and a title-only match would hijack one of them instead.
    if env._proc is not None:
        _set_target_pid(env._proc.pid)
        print(f"[human_demo] Engine PID: {env._proc.pid}", flush=True)

    hwnd = _find_openxcom_hwnd()
    if hwnd:
        buf = ctypes.create_unicode_buffer(256)
        _u32.GetWindowTextW(hwnd, buf, 256)
        print(f"[human_demo] Game window: '{buf.value}'", flush=True)
        _raise_game_window(hwnd)
    else:
        print("[human_demo] WARNING: OpenXcom window not found yet -- "
              "will be detected on your first click.", flush=True)

    action_q: "queue.Queue[int]" = queue.Queue()
    stop_event = threading.Event()

    # env._in_battlescape flips True on the engine's "battle_start" broadcast and
    # back to False when a geoscape/debrief state arrives (env.py).  Reading it
    # lets the poller record battle clicks as grid cells only — the sole form the
    # env replays in the battlescape.
    poller = threading.Thread(
        target=_poll_clicks,
        args=(action_q, stop_event, lambda: env._in_battlescape),
        daemon=True, name="click-poller",
    )
    poller.start()

    mode  = "a" if append else "w"
    ep    = 0
    step  = 0
    total = 0

    print(f"[human_demo] Recording to {out_path.resolve()}", flush=True)
    print(f"[human_demo] Click anywhere in the OpenXcom window to play.", flush=True)
    print(f"[human_demo] Press F12 or Ctrl+C to stop and save.\n", flush=True)

    try:
        with open(out_path, mode) as f:
            while not stop_event.is_set():
                try:
                    action, gx, gy = action_q.get(timeout=0.05)
                except queue.Empty:
                    # Keep env._in_battlescape current while the human is not
                    # clicking.  The flag is otherwise only updated inside
                    # env.step(0), which runs *after* a recorded click, so at the
                    # geoscape→battle transition the FIRST battlescape click — the
                    # weapon-slot click that OPENS the fire-mode menu — was polled
                    # by _poll_clicks with a stale False flag, recorded as a named
                    # (suppressed) battle action, and replayed as a no-op.  Draining
                    # pending broadcasts here flips the flag to True as soon as
                    # battle_start arrives, so every battlescape click is recorded
                    # as a replayable grid cell from the first one on.
                    # _next_geostate only reads broadcasts; it sends nothing.
                    try:
                        env._next_geostate(timeout=0.05)
                    except Exception:
                        pass
                    continue

                record = {
                    "ep":     ep,
                    "step":   step,
                    "obs":    _obs_to_dict(obs),
                    "action": action,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                total += 1

                in_battle = env._in_battlescape
                # The icon-bar hint only makes sense on the geoscape/base screens;
                # in the battlescape low clicks (hand weapons at y=172, action
                # menu, end-turn) are expected, and every battle click is recorded
                # as a replayable grid cell.
                if in_battle:
                    note = "  [BATTLE → grid]"
                elif gy >= 144:
                    note = "  [ICON BAR — soldier may not be selected]"
                else:
                    note = ""
                print(
                    f"  ep={ep}  step={step:4d}  ({gx:3d},{gy:3d})  {ACTION_NAMES[action]}{note}",
                    flush=True,
                )

                # env.step(0) = wait/observe: polls for next geostate broadcast
                # without sending anything to the bridge, so no double-click.
                env._action_window.append(action)  # inject real click so live heatmap shows it
                obs, _, terminated, truncated, _ = env.step(0)
                step += 1

                if terminated or truncated:
                    ep += 1
                    step = 0
                    print("[human_demo] Episode ended -- resetting ...", flush=True)
                    if n_episodes and ep >= n_episodes:
                        print(f"[human_demo] Reached {n_episodes} episode(s).", flush=True)
                        break
                    obs, _ = env.reset()

    except KeyboardInterrupt:
        print()
    finally:
        stop_event.set()
        poller.join(timeout=2.0)
        env.close()
        print(
            f"[human_demo] Saved {total} records across {ep + 1} episode(s) "
            f"-> {out_path.resolve()}",
            flush=True,
        )
        if total > 0:
            print(
                f"[human_demo] Run BC pretraining next:\n"
                f"  py -3 -m xcom_rl.bc_pretrain --demos {out_path} --epochs 30",
                flush=True,
            )


# -- entry point ---------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Record human XCOM gameplay for BC pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--out", default=str(_OUT_DIR / "human_demos.jsonl"),
        help="Output JSONL path (default: python/data/human_demos.jsonl)",
    )
    p.add_argument(
        "--episodes", type=int, default=0,
        help="Stop after N complete episodes (0 = run until Ctrl+C)",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Append to an existing demo file instead of overwriting it",
    )
    args = p.parse_args()
    collect(Path(args.out), n_episodes=args.episodes, append=args.append)


if __name__ == "__main__":
    main()
