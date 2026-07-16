"""
PPO training loop for the XCOM strategic-layer task.

Setup
-----
  N_ENVS instances via SB3's SubprocVecEnv (one process per env):
    rank 0          : non-headless visible ("spectator"), speed_ms=10.
    ranks 1..N-1    : headless, prefill_ufo=True.

  Each instance binds to a unique AF-UNIX socket
  (openxcom_bridge_{rank}.sock inside %TEMP%).

Policy
------
  SB3 MultiInputPolicy with a custom XcomExtractor:
    economy  branch : [funds, score, total_score]                      = 3 dims
                                              → Linear(3,  16) → ReLU
    time     branch : [month, day]                                     = 2 dims
                                              → Linear(2,  8)  → ReLU
    mission  branch : [craft_status(3) + craft_fuel_pct(1) +
                       craft_damage_pct(1) + ufo_detected(2) +
                       mission_phase(4) + popup_depth(1) +
                       ending(3) + num_soldiers(1)]                    = 16 dims
                                              → Linear(16, 32) → ReLU
    fusion          : cat(16+8+32=56)         → Linear(56, 64) → ReLU

  SB3 preprocesses Discrete(n) sub-spaces to one-hot (B, n) float32 tensors
  before forwarding to the extractor, so each branch receives float tensors.

Reward
------
  total_score_delta * 0.001  (continuous strategic feedback per step)
  + phase bonus (+10 ufo_visible, +50 craft_launched)
  + cydonia bonus (+1000 if game won)
  - 0.01 step penalty

Logging
-------
  TensorBoard: tensorboard --logdir runs/

Usage
-----
  # single-env:
  py -3 -m xcom_rl.train_ppo --n-envs 1

  # multi-instance:
  py -3 -m xcom_rl.train_ppo --n-envs 9 --timesteps 100000

  # all visible:
  py -3 -m xcom_rl.train_ppo --n-envs 8 --all-visual --timesteps 50000
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

# Make sure sibling packages are importable when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401 — triggers gymnasium.register
from xcom_gym.env import XcomInterceptionEnv

# ── training constants (overrideable via CLI) ──────────────────────────────────

N_ENVS             = 9        # 1 spectator + 8 headless  (use 1 without engine rebuild)
SPECTATOR_SPEED_MS = 10       # ms/tick for the visible instance (0 = ~80 ms engine default)
MAX_STEPS          = 3_000    # per-episode truncation (headless + prefill; UFO always present)
MAX_STEPS_VISUAL   = 44_640   # per-episode truncation for all-visual mode: 31 in-game days
                               # 31 days × 1 440 steps/day (1 step = 1 min = 12 × 5-sec ticks)
TOTAL_TIMESTEPS    = 100_000
LOG_DIR            = "./runs/"
CKPT_DIR           = "./checkpoints/"
CKPT_FREQ          = 10_000   # save every N total timesteps

# PPO hyperparameters
N_STEPS    = 256    # rollout length per env per update cycle
BATCH_SIZE = 64     # minibatch size for gradient updates
N_EPOCHS   = 4      # passes over the rollout buffer per update
GAMMA      = 0.99
LR         = 3e-4
ENT_COEF   = 0.10   # entropy bonus — higher keeps exploration alive for bandit-like tasks


# ── custom feature extractor ───────────────────────────────────────────────────

class XcomExtractor(BaseFeaturesExtractor):
    """
    Three-branch encoder matching the heterogeneous Dict observation space.

    SB3 preprocesses the Dict obs before calling forward():
      Box(shape=(1,), dtype=int32) sub-spaces → (B, 1) float32
      Discrete(n) sub-spaces                  → (B, n) float32 one-hot

    Branch inputs after preprocessing:
      economy  : [funds(1), score(1), total_score(1)]               = 3 dims
      time     : [month(1), day(1)]                                 = 2 dims
      mission  : [craft_status(3), craft_fuel_pct(1),
                  craft_damage_pct(1), ufo_detected(2),
                  mission_phase(4), popup_depth(1),
                  ending(3), num_soldiers(1)]                       = 16 dims
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)

        self.economy_branch = nn.Sequential(nn.Linear(3,  16), nn.ReLU())
        self.time_branch    = nn.Sequential(nn.Linear(2,  8),  nn.ReLU())
        self.mission_branch = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.fusion         = nn.Sequential(nn.Linear(16 + 8 + 32, features_dim), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # SB3 casts int32 Box obs to float32 but does NOT scale them.
        # During rollout collection Discrete obs are stored as (B, 1) integers in
        # the rollout buffer; after one-hot they become (B, 1, n) — 3D — while Box
        # obs remain (B, 1) — 2D.  reshape(B, -1) flattens both to 2D uniformly.
        B = obs["funds"].shape[0]

        funds_n       = obs["funds"].reshape(B, 1)            / 1e8    # [0, 1]
        score_n       = obs["score"].reshape(B, 1)            / 1e4    # [-1, 1]
        total_score_n = obs["total_score"].reshape(B, 1)      / 1e5    # [-1, 1]
        month_n       = obs["month"].reshape(B, 1)            / 12.0   # [0.08, 1]
        day_n         = obs["day"].reshape(B, 1)              / 31.0   # [0.03, 1]
        fuel_n        = obs["craft_fuel_pct"].reshape(B, 1)   / 100.0  # [0, 1]
        damage_n      = obs["craft_damage_pct"].reshape(B, 1) / 100.0  # [0, 1]
        popup_n       = obs["popup_depth"].reshape(B, 1)      / 10.0   # [0, 1]
        soldiers_n    = obs["num_soldiers"].reshape(B, 1)     / 50.0   # [0, 1]

        economy  = self.economy_branch(torch.cat([funds_n, score_n, total_score_n], dim=1))
        time_out = self.time_branch(torch.cat([month_n, day_n], dim=1))
        mission  = self.mission_branch(torch.cat([
            obs["craft_status"].reshape(B, -1),    # (B,3)
            fuel_n,
            damage_n,
            obs["ufo_detected"].reshape(B, -1),    # (B,2)
            obs["mission_phase"].reshape(B, -1),   # (B,4)
            popup_n,
            obs["ending"].reshape(B, -1),          # (B,3)
            soldiers_n,
        ], dim=1))
        return self.fusion(torch.cat([economy, time_out, mission], dim=1))


# ── env factory ────────────────────────────────────────────────────────────────

def make_env(
    rank: int,
    n_envs: int,
    seed: int = 0,
    all_visual: bool = False,
) -> Callable[[], gym.Env]:
    """
    Returns a factory (closure) that SubprocVecEnv calls in each worker process.

    Normal mode (all_visual=False):
      rank 0          : visible spectator (headless=False, speed_ms=SPECTATOR_SPEED_MS)
      ranks 1..N-1    : headless training envs with UFO prefill

    All-visual mode (all_visual=True):
      all ranks       : visible (headless=False, speed_ms=SPECTATOR_SPEED_MS)
                        UFO prefill is skipped (requires headless); episodes start
                        at game-day 1 and wait for a natural UFO spawn.
    """
    def _init() -> gym.Env:
        if all_visual:
            is_headless = False
            speed       = SPECTATOR_SPEED_MS
        else:
            is_headless = not (rank == 0 and n_envs > 1)
            speed       = SPECTATOR_SPEED_MS if (rank == 0 and n_envs > 1) else 0
        sock = "openxcom_bridge.sock" if n_envs == 1 else f"openxcom_bridge_{rank}.sock"
        env = XcomInterceptionEnv(
            max_steps   = MAX_STEPS_VISUAL if all_visual else MAX_STEPS,
            headless    = is_headless,
            socket_name = sock,
            speed_ms    = speed,
            prefill_ufo = True,   # only active when headless=True
        )
        return env

    set_random_seed(seed + rank)
    return _init


# ── main training loop ─────────────────────────────────────────────────────────

def _load_bc_weights(model: PPO, bc_path: str) -> None:
    """Inject BC-pretrained policy weights into *model* before PPO rollouts begin."""
    zip_path = bc_path if bc_path.endswith(".zip") else bc_path + ".zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("policy.pth") as fh:
            state = torch.load(io.BytesIO(fh.read()), map_location=model.device)
    model.policy.load_state_dict(state)
    print(f"[train_ppo] Loaded BC weights from {zip_path}")


def train(
    n_envs: int,
    total_timesteps: int,
    log_dir: str,
    all_visual: bool = False,
    bc_weights: Optional[str] = None,
    run_name: str = "PPO",
) -> None:
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n[train_ppo] n_envs={n_envs}  total_timesteps={total_timesteps}")
    if all_visual:
        print(f"  all {n_envs} envs : visible (speed_ms={SPECTATOR_SPEED_MS}, no UFO prefill)")
    elif n_envs > 1:
        print(f"  rank 0 : visible spectator (speed_ms={SPECTATOR_SPEED_MS})")
        print(f"  ranks 1..{n_envs-1} : headless training envs")

    env_fns = [make_env(rank=i, n_envs=n_envs, all_visual=all_visual)
               for i in range(n_envs)]
    if n_envs == 1:
        # SubprocVecEnv requires at least 2 envs on some platforms; use DummyVecEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor"))
    # VecNormalize: norm_obs=False to avoid corrupting Discrete sub-spaces (see
    # XcomExtractor for manual scaling).  norm_reward=False because shaped_reward
    # episodes all return ~9.99, so z-score normalization collapses advantages to
    # near-zero and gradient noise dominates — the policy drifts to action=0 and
    # never recovers.  Raw shaped rewards are already in a sensible LR range.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    # Policy kwargs: inject our custom feature extractor
    policy_kwargs = dict(
        features_extractor_class  = XcomExtractor,
        features_extractor_kwargs = dict(features_dim=64),
        net_arch                  = dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        policy         = "MultiInputPolicy",
        env            = vec_env,
        n_steps        = N_STEPS,
        batch_size     = BATCH_SIZE,
        n_epochs       = N_EPOCHS,
        gamma          = GAMMA,
        learning_rate  = LR,
        ent_coef       = ENT_COEF,
        policy_kwargs  = policy_kwargs,
        tensorboard_log= log_dir,
        verbose        = 1,
    )

    print(f"\n[train_ppo] Policy:\n{model.policy}\n")

    if bc_weights is not None:
        _load_bc_weights(model, bc_weights)

    ckpt_prefix = f"xcom_ppo_{run_name.lower().replace(' ', '_')}"
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CKPT_FREQ // n_envs, 1),
        save_path   = CKPT_DIR,
        name_prefix = ckpt_prefix,
    )

    print(f"[train_ppo] Training for {total_timesteps} timesteps  run_name={run_name} ...")
    print(f"  TensorBoard: tensorboard --logdir {os.path.abspath(log_dir)}\n")

    model.learn(
        total_timesteps = total_timesteps,
        callback        = checkpoint_cb,
        log_interval    = 1,
        progress_bar    = False,
        tb_log_name     = run_name,
    )

    save_path = os.path.join(CKPT_DIR, f"xcom_ppo_{run_name.lower().replace(' ', '_')}_final")
    model.save(save_path)
    # Save VecNormalize running statistics alongside the model so they can be
    # restored for evaluation (observation means/variances must match training).
    if isinstance(vec_env, VecNormalize):
        vec_env.save(os.path.join(CKPT_DIR, "vecnormalize.pkl"))
    print(f"\n[train_ppo] Training complete.  Model saved to {save_path}.zip")

    vec_env.close()


# ── entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on XcomInterception-v0")
    p.add_argument("--n-envs",      type=int, default=N_ENVS,
                   help="Total env instances (1 = no spectator, no rebuild needed)")
    p.add_argument("--timesteps",   type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--log-dir",     type=str, default=LOG_DIR)
    p.add_argument("--all-visual",  action="store_true",
                   help="Open a visible window for every env (no headless/prefill)")
    p.add_argument("--bc-weights",  type=str, default=None,
                   help="Path to BC-pretrained weights (.zip) to initialise policy before PPO")
    p.add_argument("--run-name",    type=str, default=None,
                   help="TensorBoard run name (default: 'bc_init' if --bc-weights else 'cold_start')")
    return p.parse_args()


if __name__ == "__main__":
    # Guard required for SubprocVecEnv on Windows (spawn start method)
    args = _parse_args()
    if args.run_name is None:
        args.run_name = "bc_init" if args.bc_weights else "cold_start"
    train(
        n_envs          = args.n_envs,
        total_timesteps = args.timesteps,
        log_dir         = args.log_dir,
        all_visual      = args.all_visual,
        bc_weights      = args.bc_weights,
        run_name        = args.run_name,
    )
