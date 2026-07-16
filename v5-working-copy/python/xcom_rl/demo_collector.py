"""
Collect (obs, action) demonstration pairs for behaviour-cloning pretraining.

Expert click policy:
  geoscape (no popup)  → speed_1day (fast-forward to find UFOs)
  ufo_detected popup   → ufo_intercept
  intercept screen     → intercept_craft1
  confirm_dest dialog  → confirm_ok
  mission_detected     → mission_cancel  (skip terror missions for simplicity)
  info dialogs         → dismiss         (monthly_report / low_fuel / etc.)

The demo collector runs N episodes of at most MAX_STEPS steps each, recording
every (obs, action) pair.  Each episode resets the engine via the new_game
bridge command (no engine restart required).

Usage:
    py -3 -m xcom_rl.demo_collector
    py -3 -m xcom_rl.demo_collector --episodes 30 --steps 3000
    py -3 -m xcom_rl.demo_collector --out python/data/demos.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import (
    XcomInterceptionEnv,
    ACTIONS,
    ACTION_NAMES,
    SCREEN_TYPES,
)

_ROOT    = Path(__file__).parent.parent.parent
_OUT_DIR = _ROOT / "python" / "data"

_SCREEN_IDX_TO_NAME = {v: k for k, v in SCREEN_TYPES.items()}


# ── helpers ────────────────────────────────────────────────────────────────────

def _screen(obs: Dict[str, Any]) -> str:
    return _SCREEN_IDX_TO_NAME.get(int(obs["screen_type"]), "other")


def _obs_to_dict(obs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = int(v)
    return out


# ── expert policy ──────────────────────────────────────────────────────────────

def _expert_action(obs: Dict[str, Any]) -> int:
    """
    Deterministic expert that navigates the Geoscape via GUI clicks.
    Returns an action index from ACTION_NAMES.
    """
    screen = _screen(obs)

    if screen == "ufo_detected":
        return ACTION_NAMES.index("ufo_intercept")

    if screen == "intercept":
        return ACTION_NAMES.index("intercept_craft1")

    if screen == "confirm_dest":
        return ACTION_NAMES.index("confirm_ok")

    if screen == "mission_detected":
        return ACTION_NAMES.index("mission_cancel")

    if screen in ("low_fuel", "craft_error", "monthly_report", "items_arriving", "craft_patrol"):
        return ACTION_NAMES.index("dismiss")

    # Geoscape — fast-forward time to trigger events
    return ACTION_NAMES.index("speed_1day")


# ── collection loop ────────────────────────────────────────────────────────────

def collect(
    n_episodes: int,
    max_steps_per_ep: int,
    out_path: Path,
    socket_suffix: str = "demo0",
) -> None:
    env = XcomInterceptionEnv(
        headless    = True,
        socket_name = f"openxcom_bridge_{socket_suffix}.sock",
        max_steps   = max_steps_per_ep,
    )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_records  = 0
    action_counts: Dict[int, int] = {}
    launches       = 0
    t0             = time.time()

    try:
        with open(out_path, "w") as f:
            for ep in range(n_episodes):
                obs, _ = env.reset()
                for step in range(max_steps_per_ep):
                    action = _expert_action(obs)
                    record = {
                        "ep":     ep,
                        "step":   step,
                        "obs":    _obs_to_dict(obs),
                        "action": action,
                    }
                    f.write(json.dumps(record) + "\n")
                    total_records += 1
                    action_counts[action] = action_counts.get(action, 0) + 1

                    if ACTION_NAMES[action] == "confirm_ok":
                        launches += 1

                    nxt, _, terminated, truncated, _ = env.step(action)
                    obs = nxt
                    if terminated or truncated:
                        break

                if (ep + 1) % 5 == 0:
                    elapsed = time.time() - t0
                    print(f"  ep {ep+1}/{n_episodes}  records={total_records}"
                          f"  launches={launches}  {elapsed:.0f}s", flush=True)
    finally:
        env.close()

    dt = time.time() - t0
    named_counts = {ACTION_NAMES[k]: v for k, v in sorted(action_counts.items())}
    print(f"\n[demo_collector] Done in {dt:.1f}s", flush=True)
    print(f"  total records : {total_records}", flush=True)
    print(f"  launches      : {launches}", flush=True)
    print(f"  action dist   : {named_counts}", flush=True)
    print(f"  saved to      : {out_path.resolve()}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Collect BC demo dataset (click-based)")
    p.add_argument("--episodes",  type=int, default=30,
                   help="Number of full episodes to collect")
    p.add_argument("--steps",     type=int, default=5000,
                   help="Max steps per episode")
    p.add_argument("--out",       type=str,
                   default=str(_OUT_DIR / "demos.jsonl"))
    args = p.parse_args()

    print(f"[demo_collector] Collecting {args.episodes} episodes "
          f"(max {args.steps} steps each) …", flush=True)
    collect(
        n_episodes       = args.episodes,
        max_steps_per_ep = args.steps,
        out_path         = Path(args.out),
    )


if __name__ == "__main__":
    main()
