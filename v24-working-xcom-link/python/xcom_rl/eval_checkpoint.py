"""
eval_checkpoint.py — evaluate a PPO checkpoint and record results in the registry.

Can be imported as a module (eval_and_register) or run as a CLI.

Usage
-----
  # Evaluate a specific checkpoint
  py -3 -m xcom_rl.eval_checkpoint --checkpoint checkpoints/xcom_ppo_run1_50000_steps.zip

  # Evaluate the current best checkpoint (by composite score)
  py -3 -m xcom_rl.eval_checkpoint --checkpoint best --n-episodes 40

  # List current rankings without running new episodes
  py -3 -m xcom_rl.eval_checkpoint --list

  # Save high-reward episodes as demo data for BC pretraining
  py -3 -m xcom_rl.eval_checkpoint --checkpoint best --save-demos --demo-threshold 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from xcom_rl.checkpoint_registry import CheckpointRegistry

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import XcomInterceptionEnv

# Import constants from train_ppo to stay consistent
from xcom_rl.train_ppo import MAX_STEPS, MAX_STEPS_VISUAL, SPECTATOR_SPEED_MS, CKPT_DIR


_EVAL_RANK_OFFSET = 100  # eval envs use ranks 100+ so logs don't collide with training envs

def _make_eval_env(rank: int, max_steps: int = MAX_STEPS):
    """Factory for a single headless eval env with a unique socket name."""
    def _init() -> XcomInterceptionEnv:
        return XcomInterceptionEnv(
            headless=True,
            speed_ms=0,
            socket_name=f"openxcom_bridge_{_EVAL_RANK_OFFSET + rank}.sock",
            max_steps=max_steps,
        )
    return _init


def eval_and_register(
    ckpt_path: str,
    n_episodes: int,
    n_envs: int,
    registry: "CheckpointRegistry",
    save_demos: bool = False,
    demo_threshold: float = 150.0,
    demo_out: Optional[str] = None,
    max_steps: int = MAX_STEPS,
) -> None:
    """
    Load *ckpt_path*, run *n_episodes* of evaluation across *n_envs* headless
    envs, record metrics in *registry*, and optionally save high-quality
    episodes to *demo_out* for BC pretraining.

    This function creates fresh engine processes (always headless) and closes
    them when done. Safe to call after training envs have been closed.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    env_fns = [_make_eval_env(i, max_steps=max_steps) for i in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    try:
        model = PPO.load(ckpt_path, env=vec_env)
    except Exception as e:
        print(f"[eval] Failed to load {ckpt_path}: {e}")
        vec_env.close()
        return

    norm_path = os.path.join(os.path.dirname(ckpt_path), "vecnormalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    obs = vec_env.reset()

    ep_data: list[dict] = []
    current_ep_demos: list[list] = [[] for _ in range(n_envs)]

    print(f"[eval] {os.path.basename(ckpt_path)}: {n_episodes} episodes × {n_envs} envs …")
    t0 = time.time()

    while len(ep_data) < n_episodes:
        actions, _ = model.predict(obs, deterministic=False)

        if save_demos:
            for i in range(n_envs):
                obs_dict = {k: obs[k][i].tolist() for k in obs}
                current_ep_demos[i].append((obs_dict, int(actions[i])))

        obs, _rewards, dones, _infos = vec_env.step(actions)

        for i in range(n_envs):
            if dones[i] and len(ep_data) < n_episodes:
                ep_info_list = vec_env.get_attr("last_ep_info", indices=[i])
                ep_info = ep_info_list[0] if ep_info_list else {}
                ep_data.append(ep_info)

                ep_reward = ep_info.get("reward", 0.0)
                if save_demos and current_ep_demos[i] and ep_reward >= demo_threshold:
                    _flush_demos(current_ep_demos[i], demo_out, ep_reward)
                current_ep_demos[i] = []

    elapsed = time.time() - t0
    print(f"[eval] Done in {elapsed:.1f}s  ({len(ep_data)} episodes)")

    vec_env.close()

    metrics = _compute_metrics(ep_data, n_envs)
    registry.register_eval(ckpt_path, metrics)
    registry.save()
    print()
    registry.print_table()


def _compute_metrics(ep_data: list[dict], n_envs: int) -> dict:
    n = max(len(ep_data), 1)

    rewards = [ep.get("reward",      0.0) for ep in ep_data]
    months  = [ep.get("month",       1)   for ep in ep_data]
    scores  = [ep.get("total_score", 0)   for ep in ep_data]
    endings = [ep.get("ending",      0)   for ep in ep_data]
    steps   = [ep.get("steps",       0)   for ep in ep_data]

    def milestone_rate(key: str) -> float:
        return sum(
            1 for ep in ep_data
            if ep.get("tally", {}).get(key, {}).get("n", 0) > 0
        ) / n

    return {
        "n_episodes":         n,
        "n_envs":             n_envs,
        "mean_reward":        round(float(np.mean(rewards)),  3),
        "std_reward":         round(float(np.std(rewards)),   3),
        "max_reward":         round(float(max(rewards)),      3),
        "min_reward":         round(float(min(rewards)),      3),
        "win_rate":           round(sum(1 for e in endings if e == 1) / n, 4),
        "loss_rate":          round(sum(1 for e in endings if e == 2) / n, 4),
        "truncation_rate":    round(sum(1 for e in endings if e == 0) / n, 4),
        "mean_month_reached": round(float(np.mean(months)),   3),
        "max_month_reached":  int(max(months)),
        "mean_total_score":   round(float(np.mean(scores)),   3),
        "mean_steps_per_ep":  round(float(np.mean(steps)),    1),
        "craft_launch_rate":  round(milestone_rate("craft_launch"),  4),
        "research_rate":      round(milestone_rate("research_start"), 4),
        "build_rate":         round(milestone_rate("build"),         4),
        "shoot_down_rate":    round(milestone_rate("shoot_down"),    4),
        "alien_base_rate":    round(milestone_rate("alien_base"),    4),
        "manufacture_rate":   round(milestone_rate("mfg_start"),     4),
        "new_base_rate":      round(milestone_rate("new_base"),      4),
    }


def _flush_demos(steps: list, demo_out: Optional[str], ep_reward: float) -> None:
    out_path = demo_out or os.path.join("data", "demos.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a") as f:
        for obs_dict, action in steps:
            f.write(json.dumps({"obs": obs_dict, "action": action}) + "\n")
    print(f"[eval] Saved {len(steps)} demo steps (ep_reward={ep_reward:.1f}) → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO checkpoint and update the registry")
    p.add_argument("--checkpoint",      type=str, default=None,
                   help="Path to .zip, or 'best' to use the registry's top-ranked checkpoint")
    p.add_argument("--n-episodes",      type=int, default=20)
    p.add_argument("--n-envs",          type=int, default=4)
    p.add_argument("--registry",        type=str, default=os.path.join(CKPT_DIR, "registry.json"))
    p.add_argument("--select-by",       type=str, default="composite",
                   choices=["composite", "reward"],
                   help="Criterion when --checkpoint best")
    p.add_argument("--save-demos",      action="store_true",
                   help="Append high-reward episodes to data/demos.jsonl for BC pretraining")
    p.add_argument("--demo-threshold",  type=float, default=150.0,
                   help="Minimum episode reward to save as a demo (default: 150)")
    p.add_argument("--demo-out",        type=str, default=None,
                   help="Demo output path (default: data/demos.jsonl)")
    p.add_argument("--list",            action="store_true",
                   help="Print current registry rankings and exit (no evaluation)")
    return p.parse_args()


if __name__ == "__main__":
    from xcom_rl.checkpoint_registry import CheckpointRegistry

    args = _parse_args()

    registry = CheckpointRegistry(args.registry)
    registry.load()

    if args.list:
        registry.print_table()
        sys.exit(0)

    if args.checkpoint is None:
        print("[eval] Provide --checkpoint <path> or --checkpoint best, or use --list")
        sys.exit(1)

    if args.checkpoint == "best":
        ckpt = registry.get_best(by=args.select_by)
        if ckpt is None:
            print(f"[eval] No evaluated checkpoints in registry. "
                  f"Provide an explicit --checkpoint path.")
            sys.exit(1)
        print(f"[eval] Best checkpoint (by {args.select_by}): {ckpt}")
    else:
        ckpt = args.checkpoint

    if not os.path.exists(ckpt):
        print(f"[eval] Checkpoint not found: {ckpt}")
        sys.exit(1)

    eval_and_register(
        ckpt_path      = ckpt,
        n_episodes     = args.n_episodes,
        n_envs         = args.n_envs,
        registry       = registry,
        save_demos     = args.save_demos,
        demo_threshold = args.demo_threshold,
        demo_out       = args.demo_out,
    )
