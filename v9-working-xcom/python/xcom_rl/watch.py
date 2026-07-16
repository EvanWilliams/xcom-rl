"""
Live training monitor — run in a separate terminal alongside train_ppo.py.

  py -3 -m xcom_rl.watch
  train.bat watch

Reads two sources per environment:
  logs/live_env_{rank}.json   — updated every step  (instantaneous reward)
  logs/env_{rank}.jsonl       — updated per episode  (completed episode stats)

Refreshes every 0.5 s. Press Ctrl-C to exit.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

REFRESH_S   = 0.5
ENDING_NAME = {0: "", 1: "WIN", 2: "LOSE"}

# ANSI helpers
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"

SCREEN_SHORT = {
    "geoscape":        "geo",
    "ufo_detected":    "UFO!",
    "mission_detected":"MISS!",
    "intercept":       "intcpt",
    "confirm_dest":    "cnfirm",
    "low_fuel":        "fuel",
    "craft_error":     "err",
    "monthly_report":  "month",
    "items_arriving":  "items",
    "craft_patrol":    "patrol",
    "other":           "?",
    "main_menu":       "MENU",
}


def _color_r(r: float) -> str:
    s = f"{r:+8.3f}"
    if r > 5:
        return GREEN + BOLD + s + RESET
    if r > 0:
        return GREEN + s + RESET
    if r < -5:
        return RED + BOLD + s + RESET
    if r < 0:
        return RED + s + RESET
    return DIM + s + RESET


def _color_score(s: int) -> str:
    if s > 0:
        return GREEN + f"+{s}" + RESET
    if s < 0:
        return RED + str(s) + RESET
    return str(s)


def read_live(rank: int) -> Optional[Dict]:
    path = os.path.join(_LOG_DIR, f"live_env_{rank}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_episodes(path: str) -> List[Dict]:
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


def render(ranks: List[int], ep_data: Dict[int, List[Dict]], start_time: float) -> None:
    now     = time.time()
    elapsed = now - start_time

    print(CLEAR, end="")
    print(BOLD + CYAN + " XCOM Training Monitor" + RESET +
          f"  [{datetime.now().strftime('%H:%M:%S')}  "
          f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s]")

    # Aggregate stats from completed episodes
    all_ep_rewards: List[float] = []
    total_eps   = 0
    total_steps = 0
    for eps in ep_data.values():
        all_ep_rewards.extend(e["reward"] for e in eps)
        total_eps   += len(eps)
        total_steps += sum(e["steps"] for e in eps)

    print(BOLD + f"  Total episodes: {total_eps}  |  Total steps: {total_steps:,}" + RESET)
    if all_ep_rewards:
        mean_all = sum(all_ep_rewards) / len(all_ep_rewards)
        print(f"  Mean ep reward: {_color_r(mean_all)}  "
              f"Best: {_color_r(max(all_ep_rewards))}  "
              f"Worst: {_color_r(min(all_ep_rewards))}")
    print()

    # Per-env table
    hdr = (f"  {'Env':>3}  {'StepRew':>9}  {'EpRew':>9}  {'EpStep':>7}  "
           f"{'Screen':<8}  {'Score':>6}  {'Eps':>4}  {'LastEp':>9}  {'10ep':>9}")
    print(BOLD + hdr + RESET)
    print("  " + "-" * 82)

    for rank in ranks:
        live = read_live(rank)
        eps  = ep_data.get(rank, [])

        # Live columns
        if live:
            age      = now - live.get("t", now)
            stale    = age > 10.0
            step_r   = _color_r(live["step_reward"]) if not stale else DIM + "  stale " + RESET
            ep_r     = _color_r(live["ep_reward"])   if not stale else DIM + "    -  " + RESET
            ep_step  = f"{live['ep_step']:>7,}"      if not stale else DIM + "      -" + RESET
            screen   = SCREEN_SHORT.get(live.get("screen", ""), live.get("screen", "?"))
            score    = _color_score(live.get("total_score", 0))
        else:
            step_r  = DIM + "   wait " + RESET
            ep_r    = DIM + "   wait " + RESET
            ep_step = DIM + "      -" + RESET
            screen  = "-"
            score   = "-"

        # Episode history columns
        if eps:
            last_r  = _color_r(eps[-1]["reward"])
            recent  = [e["reward"] for e in eps[-10:]]
            mean_r  = _color_r(sum(recent) / len(recent))
            n_eps   = f"{len(eps):>4}"
        else:
            last_r  = DIM + "      -" + RESET
            mean_r  = DIM + "      -" + RESET
            n_eps   = DIM + "   -" + RESET

        print(f"  {rank:>3}  {step_r}  {ep_r}  {ep_step}  "
              f"{screen:<8}  {score:>14}  {n_eps}  {last_r}  {mean_r}")

    print()

    # Recent episode feed (last 10 across all envs)
    all_flat = [e for eps in ep_data.values() for e in eps]
    all_flat.sort(key=lambda e: e["t"])
    recent_eps = all_flat[-10:]

    if recent_eps:
        print(BOLD + "  Recent episodes:" + RESET)
        print("  " + "-" * 82)
        for e in reversed(recent_eps):
            age     = now - e["t"]
            r_s     = _color_r(e["reward"])
            sc_s    = _color_score(e["total_score"])
            end_s   = ENDING_NAME.get(e.get("ending", 0), "")
            end_col = (GREEN + BOLD + end_s + RESET if end_s == "WIN"
                       else RED + end_s + RESET if end_s else "")
            trunc   = DIM + "(trunc)" + RESET if e.get("truncated") else ""
            print(f"  env{e['env']} ep{e['ep']:>3}  "
                  f"reward={r_s}  steps={e['steps']:>6,}  "
                  f"score={sc_s:>12}  "
                  f"{e.get('month',1)}/{e.get('day',1):02d}  "
                  f"{end_col}{trunc}  {DIM}{int(age):>3}s ago{RESET}")

    print()
    print(DIM + f"  Log dir: {_LOG_DIR}  |  Refresh {REFRESH_S}s  |  Ctrl-C to exit" + RESET)
    sys.stdout.flush()


def main() -> None:
    print(f"Watching {_LOG_DIR} ...")
    start_time = time.time()

    try:
        while True:
            # Discover all env ranks from both file types
            live_files = glob.glob(os.path.join(_LOG_DIR, "live_env_*.json"))
            ep_files   = glob.glob(os.path.join(_LOG_DIR, "env_*.jsonl"))

            ranks: set[int] = set()
            for p in live_files:
                try:
                    ranks.add(int(os.path.basename(p)
                                  .replace("live_env_", "").replace(".json", "")))
                except ValueError:
                    pass
            for p in ep_files:
                try:
                    ranks.add(int(os.path.basename(p)
                                  .replace("env_", "").replace(".jsonl", "")))
                except ValueError:
                    pass

            ep_data: Dict[int, List[Dict]] = {}
            for p in ep_files:
                try:
                    rank = int(os.path.basename(p)
                               .replace("env_", "").replace(".jsonl", ""))
                    ep_data[rank] = read_episodes(p)
                except ValueError:
                    pass

            if ranks:
                render(sorted(ranks), ep_data, start_time)
            else:
                print(CLEAR + "  Waiting for training to start...", end="")
                sys.stdout.flush()

            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
