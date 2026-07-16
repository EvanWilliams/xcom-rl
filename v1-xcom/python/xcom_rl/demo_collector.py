"""
Collect (obs, action) demonstration pairs for behaviour-cloning pretraining.

Expert policy for the scoped interception task:
  phase == 1 AND craft_status == 0  →  action 1  (launch)
  otherwise                          →  action 0  (wait)

Two collection passes:
  --prefill   N episodes: reset with UFO prefill → craft at launch point
              Each episode contributes 1 launch step (action=1).
  --scan      M episodes: no prefill, max K steps of waiting (action=0),
              stopped early when UFO appears (contributes 1 launch step too).

Combined dataset gives coverage of both phases.

Usage:
    py -3 -m xcom_rl.demo_collector
    py -3 -m xcom_rl.demo_collector --prefill 200 --scan 20 --scan-steps 100
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
from xcom_gym.env import XcomInterceptionEnv

_ROOT    = Path(__file__).parent.parent.parent
_OUT_DIR = _ROOT / "python" / "data"

# ── serialisation helpers ──────────────────────────────────────────────────────

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
    phase        = int(obs["mission_phase"])
    craft_status = int(obs["craft_status"])
    if phase == 1 and craft_status == 0:
        return 1   # launch
    return 0       # wait


# ── collection loops ──────────────────────────────────────────────────────────

def _collect_prefill(
    n_episodes: int,
    socket_suffix: str = "demo0",
) -> List[Dict]:
    """Start at UFO-detected point (phase=1). One launch step per episode."""
    env = XcomInterceptionEnv(
        headless    = True,
        socket_name = f"openxcom_bridge_{socket_suffix}.sock",
        prefill_ufo = True,
        max_steps   = 10,
    )
    records: List[Dict] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            action  = _expert_action(obs)
            nxt, reward, terminated, truncated, _ = env.step(action)
            records.append({
                "ep":     ep,
                "mode":   "prefill",
                "obs":    _obs_to_dict(obs),
                "action": action,
                "reward": round(float(reward), 4),
            })
            if (ep + 1) % 20 == 0:
                print(f"  prefill: {ep+1}/{n_episodes} episodes  "
                      f"records={len(records)}", flush=True)
    finally:
        env.close()
    return records


def _collect_scan(
    n_episodes: int,
    max_steps_per_ep: int,
    socket_suffix: str = "demo1",
) -> List[Dict]:
    """Start from game day 1. Collect wait steps until UFO appears, then launch."""
    env = XcomInterceptionEnv(
        headless    = True,
        socket_name = f"openxcom_bridge_{socket_suffix}.sock",
        prefill_ufo = False,
        max_steps   = max_steps_per_ep,
    )
    records: List[Dict] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            for _ in range(max_steps_per_ep):
                action = _expert_action(obs)
                records.append({
                    "ep":     ep,
                    "mode":   "scan",
                    "obs":    _obs_to_dict(obs),
                    "action": action,
                    "reward": 0.0,
                })
                nxt, reward, terminated, truncated, _ = env.step(action)
                obs = nxt
                if terminated or truncated:
                    break
            if (ep + 1) % 5 == 0:
                print(f"  scan: {ep+1}/{n_episodes} episodes  "
                      f"records={len(records)}", flush=True)
    finally:
        env.close()
    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Collect BC demo dataset")
    p.add_argument("--prefill",      type=int, default=200,
                   help="Episodes with UFO prefill (action=1 each)")
    p.add_argument("--scan",         type=int, default=20,
                   help="Episodes scanning from day 1 (mostly action=0)")
    p.add_argument("--scan-steps",   type=int, default=100,
                   help="Max steps per scan episode")
    p.add_argument("--out",          type=str,
                   default=str(_OUT_DIR / "demos.jsonl"))
    args = p.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    t0 = time.time()
    all_records: List[Dict] = []

    if args.prefill > 0:
        print(f"[demo_collector] Collecting {args.prefill} prefill episodes …", flush=True)
        all_records += _collect_prefill(args.prefill)

    if args.scan > 0:
        print(f"[demo_collector] Collecting {args.scan} scan episodes "
              f"(max {args.scan_steps} steps each) …", flush=True)
        all_records += _collect_scan(args.scan, args.scan_steps)

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    dt = time.time() - t0
    action_counts = {0: 0, 1: 0, 2: 0}
    for r in all_records:
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1

    print(f"\n[demo_collector] Done in {dt:.1f}s", flush=True)
    print(f"  total records : {len(all_records)}", flush=True)
    print(f"  action counts : {action_counts}", flush=True)
    print(f"  saved to      : {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
