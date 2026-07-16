"""
Live training monitor — run in a separate terminal alongside train_ppo.py.

  py -3 -m xcom_rl.watch

Reads python/logs/env_*.jsonl (one file per environment) and refreshes every
2 seconds, showing per-env episode stats and aggregate metrics.

Press Ctrl-C to exit.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

REFRESH_S  = 2.0
PHASE_NAME = {0: "scan", 1: "ufo", 2: "launch", 3: "return"}
ENDING_NAME = {0: "", 1: "WIN", 2: "LOSE"}

# ANSI helpers
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def color_reward(r: float) -> str:
    s = f"{r:+.1f}"
    if r > 20:
        return GREEN + BOLD + s + RESET
    if r > 0:
        return GREEN + s + RESET
    if r < -50:
        return RED + BOLD + s + RESET
    return RED + s + RESET


def color_phase(p: int) -> str:
    colors = {0: DIM, 1: YELLOW, 2: GREEN + BOLD, 3: CYAN + BOLD}
    return colors.get(p, "") + PHASE_NAME.get(p, str(p)) + RESET


def color_score(s: int) -> str:
    if s > 0:
        return GREEN + f"+{s}" + RESET
    if s < 0:
        return RED + str(s) + RESET
    return str(s)


def read_env_log(path: str) -> List[Dict]:
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return entries


def render(all_entries: Dict[int, List[Dict]], start_time: float) -> None:
    now = time.time()
    elapsed = now - start_time

    rows = []
    all_rewards: List[float] = []
    total_eps = 0
    total_steps = 0

    for rank in sorted(all_entries):
        eps = all_entries[rank]
        if not eps:
            rows.append((rank, 0, None, None, None, None, None, None))
            continue
        last = eps[-1]
        rewards = [e["reward"] for e in eps]
        all_rewards.extend(rewards)
        total_eps   += len(eps)
        total_steps += sum(e["steps"] for e in eps)

        mean_r = sum(rewards[-10:]) / len(rewards[-10:])  # rolling 10-ep mean
        rows.append((
            rank,
            len(eps),
            last["reward"],
            mean_r,
            last["steps"],
            last["phase"],
            last["total_score"],
            last.get("ending", 0),
        ))

    # Header
    w = 80
    print(CLEAR, end="")
    print(BOLD + CYAN + " XCOM Training Monitor" + RESET +
          f"  [{datetime.now().strftime('%H:%M:%S')}  elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s]")
    print(BOLD + f"  Total episodes: {total_eps}  |  Total steps: {total_steps:,}" + RESET)
    if all_rewards:
        mean_all = sum(all_rewards) / len(all_rewards)
        best     = max(all_rewards)
        worst    = min(all_rewards)
        print(f"  Mean reward (all eps): {color_reward(mean_all)}  "
              f"Best: {color_reward(best)}  "
              f"Worst: {color_reward(worst)}")
    print()

    # Column headers
    hdr = (f"  {'Env':>3}  {'Eps':>4}  {'Last Reward':>12}  "
           f"{'10-ep Mean':>11}  {'Last Steps':>10}  "
           f"{'Phase':>8}  {'Score':>8}  {'End':>5}")
    print(BOLD + hdr + RESET)
    print("  " + "─" * (w - 2))

    for row in rows:
        rank, n_eps, last_r, mean_r, steps, phase, score, ending = row
        if n_eps == 0:
            print(f"  {rank:>3}  {'—':>4}  {'waiting...':>12}")
            continue

        end_str = ENDING_NAME.get(ending, "")
        end_col = (GREEN + BOLD + end_str + RESET if end_str == "WIN"
                   else RED + end_str + RESET if end_str == "LOSE"
                   else "")

        print(
            f"  {rank:>3}  {n_eps:>4}  {color_reward(last_r):>20}  "
            f"{color_reward(mean_r):>19}  {steps:>10,}  "
            f"{color_phase(phase):>16}  {color_score(score):>16}  {end_col}"
        )

    print()

    # Recent episode feed (last 12 across all envs)
    all_flat = []
    for rank, eps in all_entries.items():
        for e in eps:
            all_flat.append(e)
    all_flat.sort(key=lambda e: e["t"])
    recent = all_flat[-12:]

    if recent:
        print(BOLD + "  Recent episodes:" + RESET)
        print("  " + "─" * (w - 2))
        for e in reversed(recent):
            age = now - e["t"]
            phase_s  = color_phase(e["phase"])
            reward_s = color_reward(e["reward"])
            score_s  = color_score(e["total_score"])
            end_s    = ENDING_NAME.get(e.get("ending", 0), "")
            end_col  = (GREEN + BOLD + end_s + RESET if end_s == "WIN"
                        else RED + end_s + RESET if end_s else "")
            trunc = DIM + "(trunc)" + RESET if e.get("truncated") else ""
            month, day = e.get("month", 1), e.get("day", 1)
            print(f"  env{e['env']} ep{e['ep']:>3}  "
                  f"reward={reward_s:>20}  steps={e['steps']:>7,}  "
                  f"phase={phase_s:>14}  score={score_s:>10}  "
                  f"game-day {month}/{day:02d}  {end_col}{trunc}  "
                  f"{DIM}{int(age):>3}s ago{RESET}")

    print()
    print(DIM + f"  Log dir: {_LOG_DIR}  |  Refresh every {REFRESH_S}s  |  Ctrl-C to exit" + RESET)
    sys.stdout.flush()


def main() -> None:
    print(f"Watching {_LOG_DIR} ...")
    start_time = time.time()

    try:
        while True:
            pattern = os.path.join(_LOG_DIR, "env_*.jsonl")
            files   = sorted(glob.glob(pattern))
            data: Dict[int, List[Dict]] = {}
            for path in files:
                try:
                    rank = int(os.path.basename(path)
                               .replace("env_", "").replace(".jsonl", ""))
                except ValueError:
                    continue
                data[rank] = read_env_log(path)

            render(data, start_time)
            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
