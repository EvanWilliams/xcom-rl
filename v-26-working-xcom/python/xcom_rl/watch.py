"""
Live training monitor — run in a separate terminal alongside train_ppo.py.

  py -3 -m xcom_rl.watch
  train.bat watch

Reads two sources per environment:
  logs/live_env_{rank}.json   — updated every step  (instantaneous state)
  logs/env_{rank}.jsonl       — updated per episode  (completed episode stats)

Refreshes every 0.5 s. Press Ctrl-C to exit.
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

REFRESH_S          = 0.5
ENDING_NAME        = {0: "", 1: "WIN", 2: "LOSE"}
_LIVE_FRESHNESS_S  = 30.0   # live files older than this are excluded from rank discovery

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
    return (CYAN + f"x{n}" + RESET) if n > 0 else DIM + " -" + RESET


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


# ── sparkline ─────────────────────────────────────────────────────────────────

_SPARK_BLOCKS = " _.:=|#X@"   # ASCII bar chart chars — low to high


def _sparkline(values: List[float], width: int = 7) -> str:
    if not values:
        return DIM + "-" * width + RESET
    vals = list(values[-width:])
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng < 1e-9:
        mid = _SPARK_BLOCKS[4]
        return (mid * len(vals)).ljust(width)
    chars = ""
    for v in vals:
        idx = int(round((v - mn) / rng * 8))
        chars += _SPARK_BLOCKS[min(max(idx, 0), 8)]
    return chars.ljust(width)


# ── file I/O ──────────────────────────────────────────────────────────────────

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


def _discover_ranks(live_files: List[str]) -> List[int]:
    """Return sorted list of ranks whose live file was updated in the last 30s."""
    _now = time.time()
    ranks: set = set()
    for p in live_files:
        try:
            r = int(os.path.basename(p).replace("live_env_", "").replace(".json", ""))
            with open(p) as _f:
                if _now - json.load(_f).get("t", 0) < _LIVE_FRESHNESS_S:
                    ranks.add(r)
        except Exception:
            pass
    return sorted(ranks)


def _run_start_time() -> float:
    path = os.path.join(_RUNS_DIR, "train_config.json")
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# ── Plan 1 — Throughput & Rollout ─────────────────────────────────────────────

def render_throughput(
    ranks: List[int],
    live_data: Dict[int, Dict],
    ep_data: Dict[int, List[Dict]],
    rank_step_hist: Dict[int, deque],
    n_steps: int,
) -> None:
    now = time.time()

    total_rate = 0.0
    slowest_rank = -1
    slowest_rate = float("inf")
    env_rates: Dict[int, float] = {}

    total_steps_all = 0
    for rank in ranks:
        eps = ep_data.get(rank, [])
        live = live_data.get(rank) or {}
        cur_steps = sum(e["steps"] for e in eps) + live.get("ep_step", 0)
        total_steps_all += cur_steps

        hist = rank_step_hist.setdefault(rank, deque(maxlen=40))
        hist.append((now, cur_steps))

        # Compute rate over last 10s window
        rate = 0.0
        for t_past, s_past in hist:
            if now - t_past <= 10.0:
                dt = now - t_past
                ds = cur_steps - s_past
                if dt > 0.5:
                    rate = max(rate, ds / dt)
                break
        env_rates[rank] = rate
        total_rate += rate
        if rate < slowest_rate and rate > 0:
            slowest_rate = rate
            slowest_rank = rank

    # Rollout fill from total live ep_step across all ranks vs expected n_steps
    live_steps_total = sum(
        (live_data.get(r) or {}).get("ep_step", 0) for r in ranks
    )
    rollout_fill = 0.0
    if n_steps > 0 and len(ranks) > 0:
        rollout_fill = min(live_steps_total / (n_steps * len(ranks)), 1.0)
    bar_width = 12
    filled = int(rollout_fill * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)  # █ / ░

    per_env = total_rate / max(len(ranks), 1)
    slowest_s = (f"  slowest env{slowest_rank} {slowest_rate:.1f}/s"
                 if slowest_rank >= 0 and len(ranks) > 1 else "")

    print(
        f"  throughput  {CYAN}{total_rate:.0f} stp/s{RESET} total"
        f"  {per_env:.1f}/env"
        f"  rollout {rollout_fill*100:.0f}% [{YELLOW}{bar}{RESET}]"
        f"{slowest_s}" + EOLN
    )


# ── Main env table ─────────────────────────────────────────────────────────────

def render_env_table(
    ranks: List[int],
    live_data: Dict[int, Dict],
    ep_data: Dict[int, List[Dict]],
) -> None:
    now = time.time()
    live_lbl = "--- Live " + "-" * 49
    hist_lbl = "--- History " + "-" * 33
    grp = f"  {'':3}  {'':1}  {live_lbl}  |  {hist_lbl}"
    hdr = (f"  {'Env':>3}  {'M':1}  {'Ep#':>4}  {'Mo/Dy':>5}  "
           f"{'Screen':<8}  {'Score':>6}  {'Step R':>8}  {'Ep R':>8}  {'Step#':>7}  |  "
           f"{'#Ep':>4}  {'Last R':>8}  {'Avg10':>8}  {'End':<4}  {'M/Dy':>5}  {'Steps':>6}")
    print(DIM + grp + RESET + EOLN)
    print(BOLD + hdr + RESET + EOLN)
    print("  " + "-" * 116 + EOLN)

    for rank in ranks:
        live = live_data.get(rank)
        eps  = ep_data.get(rank, [])

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
              f"{screen:<8}  {score}  {step_r}  {ep_r}  {ep_step}  |  "
              f"{n_eps}  {last_r}  {mean_r}  {end_s}  {last_md}  {last_steps}" + EOLN)

    print(EOLN)


# ── Recent episode feed ────────────────────────────────────────────────────────

def render_recent_feed(all_ep: List[Dict]) -> None:
    if not all_ep:
        return
    now = time.time()
    recent_eps = sorted(all_ep, key=lambda e: e["t"])[-10:]
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


# ── Reward tally ───────────────────────────────────────────────────────────────

def render_tally(ranks: List[int], live_data: Dict[int, Dict]) -> None:
    hdr = (f"  {'Env':>3}  {'nav':>6}  {'rand':>5}  {'commit':>6}  {'wait':>6}  {'score-D':>8}"
           f"  {'UFO':>3}  {'cft':>3}  {'mis':>3}  {'res':>3}  {'+misc':>5}"
           f"  {'battle':>7}  {'penalty':>8}  {'TOTAL':>9}")
    print(BOLD + "  Reward tally (lifetime):" + RESET + EOLN)
    print("  " + "-" * 111 + EOLN)
    print(BOLD + hdr + RESET + EOLN)
    print("  " + "-" * 111 + EOLN)

    for rank in ranks:
        live = live_data.get(rank)
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


# ── Plan 2 — Per-category reward sparklines ────────────────────────────────────

_TALLY_CATS = [
    ("nav",    ["click_nav"]),
    ("rand",   ["click_random"]),
    ("commit", ["click_commit"]),
    ("wait",   ["wait"]),
    ("score",  ["score_routine", "score_battle", "score_research"]),
    ("battle", ["battle"]),
    ("pen",    ["load", "abort", "options", "fac_error", "ufopaedia", "research_cancel"]),
]


def render_sparklines(ranks: List[int], ep_data: Dict[int, List[Dict]]) -> None:
    has_any = any(
        ep.get("tally") for eps in ep_data.values() for ep in eps
    )
    if not has_any:
        return

    cat_labels = [c[0] for c in _TALLY_CATS]
    col_w = 16
    hdr = "  {:>3}  ".format("#") + "  ".join(f"{lbl:<{col_w}}" for lbl in cat_labels)

    print(BOLD + "  Tally trends (last 10 eps)  [value=last ep  bar=trend _low..@high]:" + RESET + EOLN)
    print(BOLD + hdr + RESET + EOLN)

    for rank in ranks:
        eps = ep_data.get(rank, [])
        recent = [e for e in eps[-10:] if e.get("tally")]
        if not recent:
            print(f"  {rank:>3}  " + DIM + "no ep tally data" + RESET + EOLN)
            continue

        cells = []
        for _label, cats in _TALLY_CATS:
            totals = []
            for ep in recent:
                t = ep.get("tally", {})
                val = sum(t.get(c, {}).get("total", 0.0) for c in cats)
                totals.append(val)
            last_val = totals[-1] if totals else 0.0
            spark = _sparkline(totals, width=5)
            sign  = "+" if last_val >= 0 else ""
            val_s = f"{sign}{last_val:.1f}".rjust(7)
            cell  = f"{val_s} {spark}"      # 7 + 1 + 5 = 13 chars
            cells.append(f"{cell:<{col_w}}")

        print(f"  {rank:>3}  " + "  ".join(cells) + EOLN)

    print(EOLN)


# ── Plan 3 — Milestone racing board ───────────────────────────────────────────

_MILESTONES = [
    ("UFO detected",     "ufo_detect"),
    ("research started", "research_start"),
    ("craft launched",   "craft_launch"),
    ("shoot-down",       "shoot_down"),
    ("manufacture",      "mfg_start"),
    ("new base",         "new_base"),
    ("alien base",       "alien_base"),
    ("Cydonia",          "cydonia"),
]


def render_milestones(all_ep: List[Dict]) -> None:
    if not all_ep:
        return
    n_total = len(all_ep)

    print(BOLD + "  Milestones:" + RESET + EOLN)
    print(f"  {'Event':<20}  {'hit-rate':>8}  {'first hit':<22}" + EOLN)
    print("  " + "-" * 56 + EOLN)

    for label, cat in _MILESTONES:
        hit_eps = [e for e in all_ep if e.get("tally", {}).get(cat, {}).get("n", 0) > 0]
        rate = len(hit_eps) / n_total if n_total else 0.0
        rate_pct = f"{rate*100:.0f}%"

        if hit_eps:
            first = min(hit_eps, key=lambda e: (e.get("t", 0),))
            fhit = (f"env{first.get('env',0)} ep{first.get('ep',0):>3}"
                    f"  {first.get('month',1):>2}/{first.get('day',1):02d}")
            rate_col = (GREEN + f"{rate_pct:>8}" + RESET if rate > 0.5
                        else YELLOW + f"{rate_pct:>8}" + RESET if rate > 0.1
                        else f"{rate_pct:>8}")
        else:
            fhit = DIM + "-" + RESET
            rate_col = DIM + f"{rate_pct:>8}" + RESET

        print(f"  {label:<20}  {rate_col}  {fhit}" + EOLN)

    print(EOLN)


# ── Plan 4 — Game-state snapshot panel ────────────────────────────────────────

_CRAFT_STATUS_SHORT = {
    "STR_READY":      "ready",
    "STR_OUT":        "OUT",
    "STR_REPAIRS":    "repair",
    "STR_REFUELING":  "refuel",
}

def render_game_state(ranks: List[int], live_data: Dict[int, Dict]) -> None:
    has_grid = any(
        live_data.get(r, {}).get("base_grid") for r in ranks
    )
    if not has_grid:
        return

    print(BOLD + "  Game state:" + RESET + EOLN)
    hdr = (f"  {'Env':>3}  {'M':1}  {'Mo/Dy':>5}  {'Funds':>8}  "
           f"{'Craft':>7}  {'Fuel':>4}  {'Dmg':>4}  {'Grid':>6}")
    print(BOLD + hdr + RESET + EOLN)
    print("  " + "-" * 57 + EOLN)

    for rank in ranks:
        live = live_data.get(rank)
        if not live:
            print(f"  {rank:>3}  " + DIM + "no data" + RESET + EOLN)
            continue

        h      = live.get("headless")
        mode_s = (DIM + "H" + RESET) if h is True else (CYAN + "V" + RESET) if h is False else "?"
        mo_dy  = f"{live.get('month',1):>2}/{live.get('day',1):02d}"

        funds  = live.get("funds", 0)
        if funds >= 1_000_000:
            funds_s = f"${funds/1_000_000:.1f}M"
        elif funds >= 1_000:
            funds_s = f"${funds/1_000:.0f}k"
        else:
            funds_s = f"${funds}"

        craft_s  = _CRAFT_STATUS_SHORT.get(live.get("craft_status", ""), live.get("craft_status", "?")[:6])
        fuel_pct = live.get("craft_fuel_pct", 0)
        dmg_pct  = live.get("craft_damage_pct", 0)

        grid = live.get("base_grid", [0] * 36)
        # One char per row: # built, . under construction, _ empty
        row_summary = ""
        for r in range(6):
            vals = [grid[r * 6 + c] for c in range(6)]
            row_summary += "#" if any(v == 2 for v in vals) else "." if any(v == 1 for v in vals) else "_"

        fuel_c = (GREEN if fuel_pct > 50 else YELLOW if fuel_pct > 20 else RED) + f"{fuel_pct:>3}%" + RESET
        dmg_c  = (RED + f"{dmg_pct:>3}%" + RESET) if dmg_pct > 0 else DIM + "  0%" + RESET

        print(f"  {rank:>3}  {mode_s}  {mo_dy}  {funds_s:>8}  "
              f"{craft_s:>7}  {fuel_c}  {dmg_c}  {row_summary}" + EOLN)

    print(EOLN)


# ── Plan 5 — Policy health panel ──────────────────────────────────────────────

# Action category groups for aggregation
_ACT_COMMIT = frozenset({
    "ufo_intercept", "mission_intercept", "confirm_ok", "research_info_ok",
    "purchase_ok", "craft_info_ok", "craft_equip_ok", "dismantle_ok",
    "manufacture_ok", "mfg_start", "mfg_info_ok", "craft_soldiers_ok",
    "build_row0", "build_row1", "build_row2", "build_row3", "build_row4",
    "build_row5",
    "place_pos0", "place_pos1", "place_pos2", "place_pos3", "place_pos4",
    "place_pos5", "place_pos6", "place_pos7", "place_pos8", "place_pos9",
    "place_pos10", "place_pos11", "place_pos12", "place_pos13", "place_pos14",
    "place_pos15", "place_pos16", "place_pos17", "place_pos18", "place_pos19",
    "place_pos20", "place_pos21", "place_pos22", "place_pos23", "place_pos24",
    "place_pos25", "place_pos26", "place_pos27", "place_pos28", "place_pos29",
    "place_pos30", "place_pos31", "place_pos32", "place_pos33", "place_pos34",
    "place_pos35",
})
_ACT_RANDOM  = frozenset({"random_click", "random_right_click"})
_ACT_WAIT    = frozenset({"wait", "wait_2", "dismiss"})
_ACT_KEY     = frozenset({"key_a", "key_enter", "menu_new_game", "battle_pickup"})

# Screen category groups
_SCR_GEO  = frozenset({
    "geoscape", "ufo_detected", "mission_detected", "intercept", "confirm_dest",
    "low_fuel", "craft_error", "monthly_report", "items_arriving", "craft_patrol",
    "dogfight",
})
_SCR_BASE  = frozenset({"basescape", "soldiers", "soldier_info", "soldier_armor"})
_SCR_RSCH  = frozenset({"research", "research_new", "research_info"})
_SCR_CRFT  = frozenset({"crafts", "craft_info", "craft_soldiers", "craft_equipment"})
_SCR_MFG   = frozenset({"manufacture", "build_facilities", "place_facility"})
_SCR_BUY   = frozenset({"purchase", "sell"})


def render_policy_health(ranks: List[int], live_data: Dict[int, Dict]) -> None:
    has_data = any(live_data.get(r, {}).get("action_hist") for r in ranks)
    if not has_data:
        return

    # Aggregate across all envs
    act_total: Dict[str, int] = {}
    scr_total: Dict[str, int] = {}
    stuck_per_rank: Dict[int, int] = {}

    for rank in ranks:
        live = live_data.get(rank) or {}
        for k, v in live.get("action_hist", {}).items():
            act_total[k] = act_total.get(k, 0) + v
        for k, v in live.get("screen_hist", {}).items():
            scr_total[k] = scr_total.get(k, 0) + v
        stuck_per_rank[rank] = live.get("stuck_total", 0)

    act_sum = sum(act_total.values()) or 1

    def _act_pct(names) -> float:
        return sum(act_total.get(n, 0) for n in names) / act_sum * 100

    def _scr_pct(names) -> float:
        s = sum(scr_total.values()) or 1
        return sum(scr_total.get(n, 0) for n in names) / s * 100

    nav_pct    = 100 - _act_pct(_ACT_COMMIT | _ACT_RANDOM | _ACT_WAIT | _ACT_KEY)
    commit_pct = _act_pct(_ACT_COMMIT)
    rand_pct   = _act_pct(_ACT_RANDOM)
    wait_pct   = _act_pct(_ACT_WAIT)
    other_pct  = _act_pct(_ACT_KEY)

    # Action entropy (bits)
    entropy = 0.0
    for cnt in act_total.values():
        p = cnt / act_sum
        if p > 0:
            entropy -= p * math.log2(p)

    geo_pct  = _scr_pct(_SCR_GEO)
    base_pct = _scr_pct(_SCR_BASE)
    rsch_pct = _scr_pct(_SCR_RSCH)
    crft_pct = _scr_pct(_SCR_CRFT)
    mfg_pct  = _scr_pct(_SCR_MFG)
    buy_pct  = _scr_pct(_SCR_BUY)
    scr_other = max(0.0, 100 - geo_pct - base_pct - rsch_pct - crft_pct - mfg_pct - buy_pct)

    stuck_strs = []
    for r in ranks:
        s = stuck_per_rank.get(r, 0)
        if s >= 5:
            stuck_strs.append(RED + f"env{r}:{s}!" + RESET)
        elif s > 0:
            stuck_strs.append(YELLOW + f"env{r}:{s}" + RESET)
        else:
            stuck_strs.append(DIM + f"env{r}:0" + RESET)

    print(BOLD + "  Policy health (all envs, lifetime):" + RESET + EOLN)
    print(
        f"  actions  nav {nav_pct:.0f}%  commit {commit_pct:.0f}%"
        f"  random {rand_pct:.0f}%  wait {wait_pct:.0f}%  key {other_pct:.0f}%"
        f"  entropy {CYAN}{entropy:.2f}b{RESET}" + EOLN
    )
    print(
        f"  screens  geo {geo_pct:.0f}%  base {base_pct:.0f}%"
        f"  rsch {rsch_pct:.0f}%  crft {crft_pct:.0f}%"
        f"  mfg {mfg_pct:.0f}%  buy {buy_pct:.0f}%  other {scr_other:.0f}%" + EOLN
    )
    if stuck_strs:
        print("  stuck    " + "  ".join(stuck_strs) + EOLN)
    print(EOLN)


# ── Master render ──────────────────────────────────────────────────────────────

def render(
    ranks: List[int],
    live_data: Dict[int, Dict],
    ep_data: Dict[int, List[Dict]],
    all_ep: List[Dict],
    start_time: float,
    rank_step_hist: Dict[int, deque],
    n_steps: int,
    pct_str: str = "",
    eta_str: str = "",
) -> None:
    now     = time.time()
    elapsed = now - start_time

    # Capture all panel output into a buffer so we can write it atomically
    _buf      = io.StringIO()
    _real_out = sys.stdout
    sys.stdout = _buf

    print(BOLD + CYAN + " XCOM Training Monitor" + RESET +
          f"  [{datetime.now().strftime('%H:%M:%S')}  "
          f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s"
          f"{pct_str}{eta_str}]" + EOLN)

    # Summary line
    total_eps   = sum(len(eps) for eps in ep_data.values())
    total_steps = sum(sum(e["steps"] for e in eps) for eps in ep_data.values())
    all_rewards = [e["reward"] for e in all_ep]
    print(BOLD + f"  Total episodes: {total_eps}  |  Total steps: {total_steps:,}" + RESET + EOLN)
    if all_rewards:
        mean_all = sum(all_rewards) / len(all_rewards)
        print(f"  Mean ep reward: {_color_r(mean_all)}  "
              f"Best: {_color_r(max(all_rewards))}  "
              f"Worst: {_color_r(min(all_rewards))}" + EOLN)

    first_live = live_data.get(ranks[0]) if ranks else None
    if first_live and "reward_scales" in first_live:
        s = first_live["reward_scales"]
        print(f"  Reward scales:  nav={s.get('nav', 1.0):.3f}  "
              f"milestone={s.get('milestone', 1.0):.3f}  "
              f"penalty={s.get('penalty', 1.0):.3f}" + EOLN)
    print(EOLN)

    # Plan 1 — throughput
    render_throughput(ranks, live_data, ep_data, rank_step_hist, n_steps)
    print(EOLN)

    # Per-env table
    render_env_table(ranks, live_data, ep_data)

    # Recent feed
    render_recent_feed(all_ep)

    # Tally
    render_tally(ranks, live_data)

    # Plan 2 — sparklines
    render_sparklines(ranks, ep_data)

    # Plan 3 — milestones
    render_milestones(all_ep)

    # Plan 4 — game state
    render_game_state(ranks, live_data)

    # Plan 5 — policy health
    render_policy_health(ranks, live_data)

    print(DIM + f"  Log dir: {_LOG_DIR}  |  Refresh {REFRESH_S}s  |  Ctrl-C to exit" + RESET + EOLN)

    sys.stdout = _real_out

    # Clear screen, write all content, then snap cursor to top so the header
    # is visible first; user can scroll down to see lower panels.
    _real_out.write("\033[2J\033[H")
    _real_out.write(_buf.getvalue())
    _real_out.write("\033[J")   # erase any leftover content below current cursor
    _real_out.write("\033[H")   # snap viewport back to top
    _real_out.flush()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    _enable_windows_ansi()
    print(f"Watching {_LOG_DIR} ...")
    start_time = time.time()
    _step_history: deque = deque()
    _rank_step_hist: Dict[int, deque] = {}

    try:
        while True:
            run_start = _run_start_time()

            # Discover active env ranks from freshness-filtered live files
            live_files = glob.glob(os.path.join(_LOG_DIR, "live_env_*.json"))
            ranks = _discover_ranks(live_files)

            # Read all live states once (avoids double-reads in render subroutines)
            live_data: Dict[int, Dict] = {}
            for rank in ranks:
                d = read_live(rank)
                if d is not None:
                    live_data[rank] = d

            # Load episode logs, filtered to current run
            ep_files = glob.glob(os.path.join(_LOG_DIR, "env_*.jsonl"))
            ep_data: Dict[int, List[Dict]] = {}
            for p in ep_files:
                try:
                    rank = int(os.path.basename(p)
                               .replace("env_", "").replace(".jsonl", ""))
                    if rank not in ranks:
                        continue
                    ep_data[rank] = [
                        e for e in read_episodes(p) if e.get("t", 0) >= run_start
                    ]
                except ValueError:
                    pass

            all_ep = sorted(
                (e for eps in ep_data.values() for e in eps),
                key=lambda e: e["t"],
            )

            # Compute global steps done + rolling rate for ETA
            steps_done = (sum(sum(e["steps"] for e in eps) for eps in ep_data.values())
                          + sum(live_data.get(r, {}).get("ep_step", 0) for r in ranks))
            now = time.time()
            _step_history.append((now, steps_done))
            while len(_step_history) > 1 and now - _step_history[0][0] > 60.0:
                _step_history.popleft()

            cfg      = read_train_config()
            total_ts = cfg.get("total_timesteps", 0)
            n_steps  = cfg.get("n_steps", 1024)
            eta_str  = pct_str = ""
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
                render(
                    ranks, live_data, ep_data, all_ep, start_time,
                    _rank_step_hist, n_steps, pct_str, eta_str,
                )
            else:
                print(CLEAR + "  Waiting for training to start...", end="")
                sys.stdout.flush()

            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
