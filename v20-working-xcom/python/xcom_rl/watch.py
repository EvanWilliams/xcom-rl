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
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

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
CLEAR  = "\033[H"   # move to home without erasing; lines overwrite in place
EOLN   = "\033[K"   # erase to end of line — clears leftover chars on shorter lines

SCREEN_SHORT = {
    "geoscape":           "geo",
    "ufo_detected":       "UFO!",
    "mission_detected":   "MISS!",
    "intercept":          "intcpt",
    "confirm_dest":       "cnfirm",
    "low_fuel":           "fuel",
    "craft_error":        "err",
    "monthly_report":     "month",
    "items_arriving":     "items",
    "craft_patrol":       "patrol",
    "other":              "?",
    "main_menu":          "MENU",
    "basescape":          "base",
    "research_new":       "rNEW",
    "research_info":      "rINF",
    "purchase":           "PRCH",
    "sell":               "SELL",
    "soldiers":           "SLDR",
    "soldier_info":       "sINF",
    "crafts":             "CRFT",
    "craft_info":         "cINF",
    "craft_soldiers":     "cSLD",
    "craft_equipment":    "cEQP",
    "soldier_armor":      "ARMR",
    "dismantle_facility": "DISM",
    "dogfight":           "DFGT",
    "ufopaedia":          "UFOP",
    "place_facility":     "PLCE",
    "manufacture":        "MNFG",
    "build_facilities":   "BFAC",
    "research":           "RSCH",
    "briefing":           "BREF",
    "debriefing":         "DBRF",
    "statistics":         "STAT",
}


def _fmt_r(v: float, width: int = 7) -> str:
    s = f"{v:+.1f}".rjust(width)
    if v > 0:
        return GREEN + s + RESET
    if v < 0:
        return RED + s + RESET
    return DIM + s + RESET


def _fmt_n(n: int) -> str:
    return (CYAN + f"×{n}" + RESET) if n > 0 else DIM + " -" + RESET


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
    raw = f"+{s}" if s > 0 else str(s)
    padded = f"{raw:>6}"
    if s > 0:
        return GREEN + padded + RESET
    if s < 0:
        return RED + padded + RESET
    return padded


def read_live(rank: int) -> Optional[Dict]:
    path = os.path.join(_LOG_DIR, f"live_env_{rank}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_train_config() -> Dict:
    path = os.path.join(_RUNS_DIR, "train_config.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


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


def render(ranks: List[int], ep_data: Dict[int, List[Dict]], start_time: float,
           pct_str: str = "", eta_str: str = "") -> None:
    now     = time.time()
    elapsed = now - start_time

    sys.stdout.write(CLEAR)   # move cursor to top-left; no blank flash
    print(BOLD + CYAN + " XCOM Training Monitor" + RESET +
          f"  [{datetime.now().strftime('%H:%M:%S')}  "
          f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s"
          f"{pct_str}{eta_str}]" + EOLN)

    # Aggregate stats from completed episodes
    all_ep_rewards: List[float] = []
    total_eps   = 0
    total_steps = 0
    for eps in ep_data.values():
        all_ep_rewards.extend(e["reward"] for e in eps)
        total_eps   += len(eps)
        total_steps += sum(e["steps"] for e in eps)

    print(BOLD + f"  Total episodes: {total_eps}  |  Total steps: {total_steps:,}" + RESET + EOLN)
    if all_ep_rewards:
        mean_all = sum(all_ep_rewards) / len(all_ep_rewards)
        print(f"  Mean ep reward: {_color_r(mean_all)}  "
              f"Best: {_color_r(max(all_ep_rewards))}  "
              f"Worst: {_color_r(min(all_ep_rewards))}" + EOLN)

    _first_live = read_live(sorted(ranks)[0]) if ranks else None
    if _first_live and "reward_scales" in _first_live:
        s = _first_live["reward_scales"]
        print(f"  Reward scales:  nav={s.get('nav', 1.0):.3f}  "
              f"milestone={s.get('milestone', 1.0):.3f}  "
              f"penalty={s.get('penalty', 1.0):.3f}" + EOLN)
    print(EOLN)

    # Per-env table
    # live section width  = 4+2+5+2+8+2+6+2+8+2+8+2+7 = 58 chars (Ep# … Step#)
    # history section width = 4+2+8+2+8+2+4+2+5+2+6  = 45 chars (#Ep … Steps)
    live_lbl = "─── Live " + "─" * 49       # 9+49 = 58 chars
    hist_lbl = "─── History " + "─" * 33    # 12+33 = 45 chars
    grp = f"  {'':3}  {'':1}  {live_lbl}  │  {hist_lbl}"
    hdr = (f"  {'Env':>3}  {'M':1}  {'Ep#':>4}  {'Mo/Dy':>5}  "
           f"{'Screen':<8}  {'Score':>6}  {'Step R':>8}  {'Ep R':>8}  {'Step#':>7}  │  "
           f"{'#Ep':>4}  {'Last R':>8}  {'Avg10':>8}  {'End':<4}  {'M/Dy':>5}  {'Steps':>6}")
    print(DIM + grp + RESET + EOLN)
    print(BOLD + hdr + RESET + EOLN)
    print("  " + "─" * 116 + EOLN)

    for rank in ranks:
        live = read_live(rank)
        eps  = ep_data.get(rank, [])

        # Live columns
        if live:
            age     = now - live.get("t", now)
            stale   = age > 10.0
            h       = live.get("headless")
            mode_s  = (DIM + "H" + RESET) if h is True else (CYAN + "V" + RESET) if h is False else "?"
            ep_num  = f"{live['ep']:>4}"                            if not stale else DIM + "   -" + RESET
            mo_dy   = f"{live.get('month',1):>2}/{live.get('day',1):02d}" if not stale else DIM + " -/--" + RESET
            step_r  = _color_r(live["step_reward"])                 if not stale else DIM + "  stale " + RESET
            ep_r    = _color_r(live["ep_reward"])                   if not stale else DIM + "      - " + RESET
            ep_step = f"{live['ep_step']:>7,}"                      if not stale else DIM + "      -" + RESET
            screen  = SCREEN_SHORT.get(live.get("screen", ""), live.get("screen", "?"))
            score   = _color_score(live.get("total_score", 0))
        else:
            mode_s  = "?"
            ep_num  = DIM + "   -" + RESET
            mo_dy   = DIM + " -/--" + RESET
            step_r  = DIM + "    wait" + RESET
            ep_r    = DIM + "    wait" + RESET
            ep_step = DIM + "      -" + RESET
            screen  = "-"
            score   = DIM + "     -" + RESET

        # Episode history columns
        if eps:
            last_ep    = eps[-1]
            last_r     = _color_r(last_ep["reward"])
            recent     = [e["reward"] for e in eps[-10:]]
            mean_r     = _color_r(sum(recent) / len(recent))
            n_eps      = f"{len(eps):>4}"
            last_end   = ENDING_NAME.get(last_ep.get("ending", 0), "")
            if last_end == "WIN":
                end_s = GREEN + BOLD + "WIN " + RESET
            elif last_end == "LOSE":
                end_s = RED + "LOSE" + RESET
            else:
                end_s = "    "
            last_md    = f"{last_ep.get('month',1):>2}/{last_ep.get('day',1):02d}"
            last_steps = f"{last_ep.get('steps', 0):>6,}"
        else:
            last_r     = DIM + "       -" + RESET
            mean_r     = DIM + "       -" + RESET
            n_eps      = DIM + "   -" + RESET
            end_s      = "    "
            last_md    = DIM + " -/--" + RESET
            last_steps = DIM + "     -" + RESET

        print(f"  {rank:>3}  {mode_s}  {ep_num}  {mo_dy}  "
              f"{screen:<8}  {score}  {step_r}  {ep_r}  {ep_step}  │  "
              f"{n_eps}  {last_r}  {mean_r}  {end_s}  {last_md}  {last_steps}" + EOLN)

    print(EOLN)

    # Recent episode feed (last 10 across all envs)
    all_flat = [e for eps in ep_data.values() for e in eps]
    all_flat.sort(key=lambda e: e["t"])
    recent_eps = all_flat[-10:]

    if recent_eps:
        print(BOLD + "  Recent episodes:" + RESET + EOLN)
        print("  " + "-" * 82 + EOLN)
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
                  f"{end_col}{trunc}  {DIM}{int(age):>3}s ago{RESET}" + EOLN)

    print(EOLN)
    render_tally(ranks)
    print(DIM + f"  Log dir: {_LOG_DIR}  |  Refresh {REFRESH_S}s  |  Ctrl-C to exit" + RESET + EOLN)
    sys.stdout.write("\033[J")   # erase from cursor to bottom — clears leftover lines
    sys.stdout.flush()


def render_tally(ranks: List[int]) -> None:
    """Print a one-row-per-env reward tally section."""
    # Header
    hdr = (f"  {'Env':>3}  {'nav':>6}  {'rand':>5}  {'commit':>6}  {'wait':>6}  {'score-Δ':>8}"
           f"  {'UFO':>3}  {'cft':>3}  {'mis':>3}  {'res':>3}  {'+misc':>5}"
           f"  {'battle':>7}  {'penalty':>8}  {'TOTAL':>9}")
    print(BOLD + "  Reward tally (lifetime):" + RESET + EOLN)
    print("  " + "─" * 111 + EOLN)
    print(BOLD + hdr + RESET + EOLN)
    print("  " + "─" * 111 + EOLN)

    for rank in ranks:
        live = read_live(rank)
        tally = live.get("reward_tally", {}) if live else {}

        def _t(cat: str) -> dict:
            return tally.get(cat, {"n": 0, "total": 0.0})

        if not tally:
            dash = DIM + "   no data" + RESET
            print(f"  {rank:>3}  {dash}" + EOLN)
            continue

        nav    = _t("click_nav")["total"]
        rand   = _t("click_random")["total"]
        commit = _t("click_commit")["total"]
        wait   = _t("wait")["total"]
        score  = (_t("score_routine")["total"] + _t("score_battle")["total"]
                  + _t("score_research")["total"])
        ufo_n  = _t("ufo_detect")["n"]
        cft_n  = _t("craft_launch")["n"]
        mis_n  = _t("mission")["n"] + _t("shoot_down")["n"]
        res_n  = _t("research_start")["n"]
        misc_n = (_t("mfg_start")["n"] + _t("build")["n"] + _t("hire")["n"]
                  + _t("promo")["n"] + _t("new_base")["n"] + _t("alien_base")["n"]
                  + _t("cydonia")["n"] + _t("menu_escape")["n"])
        battle  = _t("battle")["total"]
        penalty = (_t("load")["total"] + _t("abort")["total"] + _t("options")["total"]
                   + _t("dismantle")["total"] + _t("fac_error")["total"]
                   + _t("ufopaedia")["total"] + _t("research_cancel")["total"])
        total   = sum(v["total"] for v in tally.values())

        print(
            f"  {rank:>3}"
            f"  {_fmt_r(nav)}"
            f"  {_fmt_r(rand, width=5)}"
            f"  {_fmt_r(commit)}"
            f"  {_fmt_r(wait)}"
            f"  {_fmt_r(score, width=8)}"
            f"  {_fmt_n(ufo_n)}"
            f"  {_fmt_n(cft_n)}"
            f"  {_fmt_n(mis_n)}"
            f"  {_fmt_n(res_n)}"
            f"  {_fmt_n(misc_n):>5}"
            f"  {_fmt_r(battle)}"
            f"  {_fmt_r(penalty, width=8)}"
            f"  {_fmt_r(total, width=9)}"
            + EOLN
        )
    print(EOLN)




def _run_start_time() -> float:
    """Return mtime of train_config.json as a proxy for when the current run started."""
    path = os.path.join(_RUNS_DIR, "train_config.json")
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def main() -> None:
    print(f"Watching {_LOG_DIR} ...")
    start_time = time.time()
    _step_history: deque = deque()   # (timestamp, steps_done)

    try:
        while True:
            run_start = _run_start_time()

            # Only show envs whose live file was updated in the last 30 s.
            # This excludes stale files from old runs and removes the old 8-env cap.
            live_files = glob.glob(os.path.join(_LOG_DIR, "live_env_*.json"))
            ep_files   = glob.glob(os.path.join(_LOG_DIR, "env_*.jsonl"))

            _now = time.time()
            ranks: set[int] = set()
            for p in live_files:
                try:
                    r = int(os.path.basename(p).replace("live_env_", "").replace(".json", ""))
                    with open(p) as _f:
                        if _now - json.load(_f).get("t", 0) < 30.0:
                            ranks.add(r)
                except Exception:
                    pass

            ep_data: Dict[int, List[Dict]] = {}
            for p in ep_files:
                try:
                    rank = int(os.path.basename(p)
                               .replace("env_", "").replace(".jsonl", ""))
                    if rank not in ranks:
                        continue
                    # Filter out episodes from previous runs
                    ep_data[rank] = [e for e in read_episodes(p) if e.get("t", 0) >= run_start]
                except ValueError:
                    pass

            # Compute steps done and rolling rate for ETA
            steps_done = sum(sum(e["steps"] for e in eps) for eps in ep_data.values())
            for rank in sorted(ranks):
                live = read_live(rank)
                if live:
                    steps_done += live.get("ep_step", 0)

            now = time.time()
            _step_history.append((now, steps_done))
            while len(_step_history) > 1 and now - _step_history[0][0] > 60.0:
                _step_history.popleft()

            cfg = read_train_config()
            total_ts = cfg.get("total_timesteps", 0)
            eta_str = pct_str = ""
            if total_ts > 0:
                pct = min(steps_done / total_ts * 100, 100.0)
                pct_str = f"  {pct:.1f}%"
                if len(_step_history) >= 2:
                    dt = _step_history[-1][0] - _step_history[0][0]
                    ds = _step_history[-1][1] - _step_history[0][1]
                    if dt > 0 and ds > 0:
                        eta_min = max(total_ts - steps_done, 0) / (ds / dt) / 60.0
                        eta_str = f"  ETA ~{eta_min:.0f}m"

            if ranks:
                render(sorted(ranks), ep_data, start_time, pct_str, eta_str)
            else:
                print(CLEAR + "  Waiting for training to start...", end="")
                sys.stdout.flush()

            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
