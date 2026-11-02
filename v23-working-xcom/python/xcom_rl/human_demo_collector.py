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
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import ACTIONS, ACTION_NAMES, XcomInterceptionEnv

_ROOT    = Path(__file__).parent.parent.parent
_OUT_DIR = _ROOT / "python" / "data"

_GAME_W          = 320      # OpenXcom internal resolution
_GAME_H          = 200
_MATCH_THRESHOLD = 25       # max game-pixel distance for a named action match

_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_VK_F12     = 0x7B

# -- window helpers ------------------------------------------------------------

_u32 = ctypes.windll.user32
_cached_hwnd: int = 0


def _find_openxcom_hwnd() -> int:
    """Return the hwnd of the visible OpenXcom window (title starts with 'OpenXcom')."""
    global _cached_hwnd
    if _cached_hwnd and _u32.IsWindow(_cached_hwnd):
        return _cached_hwnd

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
_CLICKABLE = [
    (i, x, y, btn)
    for i, (name, x, y, btn) in enumerate(ACTIONS)
    if x is not None and x >= 0 and y is not None and y >= 0
]


def _match_action(gx: int, gy: int, btn: int) -> Optional[int]:
    best_i, best_d = -1, float("inf")
    for i, x, y, b in _CLICKABLE:
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
    poll_interval: float = 0.016,   # ~60 fps
) -> None:
    """
    Poll mouse state at ~60 fps and enqueue matched actions for clicks inside
    the OpenXcom window.  No hook is installed; real OS events reach the game
    normally because the engine input filter only runs in headless mode.
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
            for pressed, prev, btn_int in ((lb, lb_prev, 1), (rb, rb_prev, 3)):
                if pressed and not prev:
                    matched = _match_action(gx, gy, btn_int)
                    if matched is not None:
                        action_q.put((matched, gx, gy))

        lb_prev, rb_prev = lb, rb
        time.sleep(poll_interval)


# -- window focus helpers ------------------------------------------------------

_HWND_TOPMOST    = ctypes.wintypes.HWND(-1)
_SWP_NOMOVE      = 0x0002
_SWP_NOSIZE      = 0x0001
_SWP_SHOWWINDOW  = 0x0040


def _raise_game_window(hwnd: int) -> None:
    """
    Put the game window above the terminal using SetWindowPos(HWND_TOPMOST).
    SetForegroundWindow fails when the terminal owns the foreground lock;
    HWND_TOPMOST bypasses that restriction without needing foreground permission.
    """
    _u32.ShowWindow(hwnd, 9)   # SW_RESTORE -- un-minimise if needed
    _u32.SetWindowPos(
        hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
        _SWP_NOMOVE | _SWP_NOSIZE | _SWP_SHOWWINDOW,
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

    env = XcomInterceptionEnv(
        headless    = False,
        socket_name = "openxcom_bridge_human.sock",
        max_steps   = 999_999,
    )

    print("[human_demo] Starting OpenXcom ...", flush=True)
    obs, _ = env.reset()

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

    poller = threading.Thread(
        target=_poll_clicks, args=(action_q, stop_event),
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

                icon_warn = "  [ICON BAR — soldier may not be selected]" if gy >= 144 else ""
                print(
                    f"  ep={ep}  step={step:4d}  ({gx:3d},{gy:3d})  {ACTION_NAMES[action]}{icon_warn}",
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
