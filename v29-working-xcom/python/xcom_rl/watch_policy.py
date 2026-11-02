"""
Policy activity heatmap — run in a separate terminal alongside train_ppo.py.

  py -3 -m xcom_rl.watch_policy

For each running environment (headless and visible) renders:
  - A 200-action frequency grid: one coloured block per action, arranged in
    5 rows of 40, coloured by viridis from purple (never used) to yellow (most used).
  - A 60-step reward timeline: one block per recent step reward, coloured
    purple (worst in window) → yellow (best in window).

Requires train_ppo.py running with env.py that logs action_window/reward_window.
Old logs without those fields display gracefully (all-purple grid).

Refreshes every 0.5 s. Press Ctrl-C to exit.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import xcom_gym  # noqa: F401
from xcom_gym.env import ACTIONS

_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
REFRESH_S = 0.5

# ── viridis colourmap ─────────────────────────────────────────────────────────
# 8 key-points sampled from matplotlib's viridis LUT; linearly interpolated.
_VIRIDIS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.000, (68,   1,  84)),
    (0.143, (72,  33, 115)),
    (0.286, (64,  67, 135)),
    (0.429, (52, 101, 141)),
    (0.571, (33, 144, 140)),
    (0.714, (39, 173, 129)),
    (0.857, (92, 200,  99)),
    (1.000, (253, 231,  37)),
]

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
CLEAR  = "\033[H"
EOLN   = "\033[K"

# Action-grid row definitions — 5 rows × 40 actions = 200 total
_ROW_LABELS = ["geo+nav", "rsrch  ", "craft  ", "mfg    ", "build  "]
_ROW_START  = [0, 40, 80, 120, 160]
_ROW_LEN    = 40   # actions per row

# Spatial heatmap grid: one cell per 3×3 game-pixel block (matches grid_* action spacing)
_GRID_W       = 107   # ceil(320/3)
_GRID_H       = 67    # ceil(200/3)
_ICON_BAR_ROW = 24    # terminal row where icon bar begins (game y=144 → cell row 48 → trow 24)

# Pre-computed (col, row) heatmap cell for each action index; None = no screen position
_ACTIONS_CELL = [
    (x // 3, y // 3)
    if (x is not None and x >= 0 and y is not None and y >= 0)
    else None
    for _, x, y, _ in ACTIONS
]


def _enable_windows_ansi() -> None:
    """Opt the Windows console into VT/ANSI escape-code processing."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode   = ctypes.c_ulong()
        ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _viridis_rgb(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for i in range(len(_VIRIDIS) - 1):
        t0, c0 = _VIRIDIS[i]
        t1, c1 = _VIRIDIS[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return (
                int(c0[0] + f * (c1[0] - c0[0])),
                int(c0[1] + f * (c1[1] - c0[1])),
                int(c0[2] + f * (c1[2] - c0[2])),
            )
    return _VIRIDIS[-1][1]


def _block(t: float) -> str:
    r, g, b = _viridis_rgb(t)
    return f"\033[38;2;{r};{g};{b}m█{RESET}"


def _rev_viridis_rgb(t: float) -> Tuple[int, int, int]:
    return _viridis_rgb(1.0 - t)   # yellow=0 (low/cold), purple=1 (high/hot)


def _action_freqs(action_window: List[int]) -> List[float]:
    counts = [0] * 200
    for a in action_window:
        if 0 <= a < 200:
            counts[a] += 1
    max_c = max(counts) if counts else 1
    if max_c == 0:
        return [0.0] * 200
    return [c / max_c for c in counts]


def _reward_norm(reward_window: List[float]) -> List[float]:
    if not reward_window:
        return []
    r_min = min(reward_window)
    r_max = max(reward_window)
    span  = r_max - r_min
    if span < 1e-9:
        return [0.5] * len(reward_window)
    return [(r - r_min) / span for r in reward_window]


def _build_heatmap(all_windows: List[List[int]]) -> List[List[int]]:
    grid = [[0] * _GRID_W for _ in range(_GRID_H)]
    for win in all_windows:
        for a in win:
            if 0 <= a < len(_ACTIONS_CELL):
                cell = _ACTIONS_CELL[a]
                if cell is not None:
                    c, r = cell
                    if 0 <= r < _GRID_H and 0 <= c < _GRID_W:
                        grid[r][c] += 1
    return grid


def _heatmap_line(top_row: List[int], bot_row: List[int], max_count: int) -> str:
    mc = max(max_count, 1)
    parts = []
    for c in range(_GRID_W):
        tt = top_row[c] / mc
        bt = bot_row[c] / mc
        tr, tg, tb = _rev_viridis_rgb(tt)
        br, bg, bb = _rev_viridis_rgb(bt)
        # ▄ lower-half block: fg = bottom half colour, bg = top half colour
        parts.append(f"\033[48;2;{tr};{tg};{tb}m\033[38;2;{br};{bg};{bb}m▄{RESET}")
    return "".join(parts)


def render_heatmap(all_windows: List[List[int]], total_steps: int) -> None:
    grid      = _build_heatmap(all_windows)
    max_count = max(grid[r][c] for r in range(_GRID_H) for c in range(_GRID_W))

    print(
        f"  {BOLD}Spatial click heatmap{RESET}  320×200  "
        f"{DIM}last {total_steps} actions across all envs  "
        f"yellow=low → purple=high{RESET}" + EOLN
    )

    empty_row = [0] * _GRID_W
    for trow in range((_GRID_H + 1) // 2):   # 34 terminal rows
        if trow == _ICON_BAR_ROW:
            sep = "─" * _GRID_W
            print(f"  {DIM}y=144 {sep}{RESET}" + EOLN)

        game_top = trow * 2
        game_bot = game_top + 1
        top_row = grid[game_top]
        bot_row = grid[game_bot] if game_bot < _GRID_H else empty_row

        label = f"{game_top * 3:3d} " if trow % 8 == 0 else "    "
        print(f"  {DIM}{label}{RESET}" + _heatmap_line(top_row, bot_row, max_count) + EOLN)

    print(EOLN)


def read_live(rank: int) -> Optional[Dict]:
    path = os.path.join(_LOG_DIR, f"live_env_{rank}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def render_env(rank: int, now: float) -> None:
    live = read_live(rank)

    # ── header ────────────────────────────────────────────────────────────────
    if live:
        age    = now - live.get("t", now)
        stale  = age > 10.0
        h      = live.get("headless")
        mode_s = (DIM + "[H]" + RESET) if h is True else (CYAN + "[V]" + RESET) if h is False else "[?]"
        ep     = live.get("ep", 0)
        mo     = live.get("month", 1)
        dy     = live.get("day",   1)
        screen = live.get("screen", "?")[:10]
        score  = live.get("total_score", 0)
        score_s = f"+{score}" if score >= 0 else str(score)
        stale_s = f"  {DIM}(stale {int(age)}s){RESET}" if stale else ""
        print(f" {BOLD}env {rank}{RESET} {mode_s}  ep:{ep:<4}  {mo:>2}/{dy:02d}  "
              f"{screen:<10}  {score_s:>6}{stale_s}" + EOLN)
    else:
        print(f" {BOLD}env {rank}{RESET} [?]  {DIM}waiting for data...{RESET}" + EOLN)

    # ── action grid ───────────────────────────────────────────────────────────
    action_window = live.get("action_window", []) if live else []
    freqs = _action_freqs(action_window)

    n_steps = len(action_window)
    print(f"  {DIM}Actions  (last {n_steps} steps):{RESET}" + EOLN)
    for row_idx, (label, start) in enumerate(zip(_ROW_LABELS, _ROW_START)):
        blocks = "".join(_block(freqs[start + i]) for i in range(_ROW_LEN))
        print(f"  {DIM}{label}{RESET} {blocks}" + EOLN)

    # ── reward timeline ───────────────────────────────────────────────────────
    reward_window = live.get("reward_window", []) if live else []
    norm_rw = _reward_norm(reward_window)
    rw_blocks = "".join(_block(t) for t in norm_rw)
    rw_pad    = " " * (60 - len(norm_rw))   # pad to fixed width when window is filling
    n_rw = len(reward_window)
    print(f"  {DIM}rewards  (last {n_rw} steps):{RESET}" + EOLN)
    print(f"  {DIM}       {RESET} {rw_blocks}{rw_pad}" + EOLN)
    print(EOLN)


def main() -> None:
    _enable_windows_ansi()
    start_time = time.time()
    print(f"Watching {_LOG_DIR} ...")

    try:
        while True:
            # Discover all env ranks from live log files
            live_files = glob.glob(os.path.join(_LOG_DIR, "live_env_*.json"))
            ranks: List[int] = []
            for p in live_files:
                try:
                    ranks.append(int(
                        os.path.basename(p).replace("live_env_", "").replace(".json", "")
                    ))
                except ValueError:
                    pass
            ranks.sort()

            now     = time.time()
            elapsed = now - start_time

            sys.stdout.write(CLEAR)
            print(
                BOLD + CYAN + " XCOM Policy Heatmap" + RESET +
                f"  [{datetime.now().strftime('%H:%M:%S')}  "
                f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s]"
                f"    {DIM}purple=0 ──→ yellow=1{RESET}" + EOLN
            )
            print(EOLN)

            if ranks:
                for rank in ranks:
                    render_env(rank, now)

                all_windows: List[List[int]] = []
                total_steps = 0
                for rank in ranks:
                    live = read_live(rank)
                    if live:
                        win = live.get("action_window", [])
                        all_windows.append(win)
                        total_steps += len(win)
                if all_windows:
                    render_heatmap(all_windows, total_steps)
            else:
                print(f"  {DIM}Waiting for training to start...{RESET}" + EOLN)

            print(DIM + f"  Log dir: {_LOG_DIR}  |  Refresh {REFRESH_S}s  |  Ctrl-C to exit" + RESET + EOLN)
            sys.stdout.write("\033[J")
            sys.stdout.flush()

            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
