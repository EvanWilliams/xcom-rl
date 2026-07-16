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
