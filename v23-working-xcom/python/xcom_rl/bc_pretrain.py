"""
Behaviour-cloning pretraining for the XCOM interception policy.

Loads one or more JSONL demo datasets (produced by human_demo_collector.py),
builds the same MultiInputPolicy / XcomExtractor architecture used by
train_ppo.py, and trains it with NLL (cross-entropy) loss via supervised
learning.

The resulting weights are saved as an SB3-compatible .zip so train_ppo.py can
load them with --bc-weights before starting PPO rollouts.

MULTI-SESSION DEMO WORKFLOW
  Record demos across multiple sessions and train on all of them:

  Session 1:
      py -3 -m xcom_rl.human_demo_collector --out python/data/demos_session1.jsonl
  Session 2 (next day, etc.):
      py -3 -m xcom_rl.human_demo_collector --out python/data/demos_session2.jsonl
  Train on both:
      py -3 -m xcom_rl.bc_pretrain --demos python/data/demos_session*.jsonl

  Alternatively, append all sessions into one file:
      py -3 -m xcom_rl.human_demo_collector --append  (uses human_demos.jsonl)
      py -3 -m xcom_rl.bc_pretrain --demos python/data/human_demos.jsonl

Usage:
    py -3 -m xcom_rl.bc_pretrain
    py -3 -m xcom_rl.bc_pretrain --demos python/data/demos.jsonl --epochs 30
    py -3 -m xcom_rl.bc_pretrain --demos python/data/s1.jsonl python/data/s2.jsonl
    py -3 -m xcom_rl.bc_pretrain --out checkpoints/bc_init
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch as th
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import XcomInterceptionEnv, N_ACTIONS
from xcom_rl.train_ppo import XcomExtractor, CKPT_DIR

_ROOT = Path(__file__).parent.parent.parent

# ── dataset helpers ────────────────────────────────────────────────────────────

# Keys present in all demo versions
_OBS_KEYS_BOX_ALWAYS = ["funds", "score", "total_score", "month", "day",
                         "craft_fuel_pct", "craft_damage_pct", "popup_depth", "num_soldiers"]
# Keys added later; absent in demos collected before the obs-space expansion
_OBS_KEYS_BOX_OPTIONAL = ["hour", "alien_bases_found"]   # shape (1,)
_OBS_KEYS_DISCRETE = ["craft_status", "ufo_detected", "mission_site", "screen_type", "ending"]

_OBS_KEYS_BOX = _OBS_KEYS_BOX_ALWAYS + _OBS_KEYS_BOX_OPTIONAL


def _load_datasets(paths: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for path in paths:
        before = len(records)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[bc_pretrain] {path}: {len(records) - before} records", flush=True)
    return records


def _build_obs_batch(records: List[Dict]) -> Dict[str, np.ndarray]:
    """Stack per-step obs dicts into numpy arrays keyed by field.

    Tolerates old demo files that are missing 'hour' or 'base_grid' — fills
    those with zeros and warns so the user knows to re-collect demos.
    """
    N = len(records)
    sample = records[0]["obs"] if records else {}

    batch: Dict[str, list] = {k: [] for k in _OBS_KEYS_BOX_ALWAYS + _OBS_KEYS_DISCRETE}
    for rec in records:
        o = rec["obs"]
        for k in _OBS_KEYS_BOX_ALWAYS:
            batch[k].append(o[k])
        for k in _OBS_KEYS_DISCRETE:
            batch[k].append(o[k])

    result: Dict[str, np.ndarray] = {}
    for k in _OBS_KEYS_BOX_ALWAYS:
        result[k] = np.array(batch[k], dtype=np.int32).reshape(-1, 1)
    for k in _OBS_KEYS_DISCRETE:
        result[k] = np.array(batch[k], dtype=np.int64)

    # Optional scalar keys added in later obs-space revisions
    for k in _OBS_KEYS_BOX_OPTIONAL:
        if k in sample:
            result[k] = np.array([rec["obs"][k] for rec in records], dtype=np.int32).reshape(-1, 1)
        else:
            result[k] = np.zeros((N, 1), dtype=np.int32)
            print(f"[bc_pretrain] WARNING: demos missing '{k}' obs — filling zeros. "
                  "Re-collect demos for accurate BC training.")

    # base_grid: shape (N, 36); added in obs-space revision after initial release
    if "base_grid" in sample:
        result["base_grid"] = np.array(
            [rec["obs"]["base_grid"] for rec in records], dtype=np.int8
        )
    else:
        result["base_grid"] = np.zeros((N, 36), dtype=np.int8)
        print("[bc_pretrain] WARNING: demos missing 'base_grid' obs — filling zeros. "
              "Re-collect demos for accurate BC training.")

    return result


# ── policy construction ────────────────────────────────────────────────────────

def _make_policy(device: str) -> PPO:
    """Build a PPO model for BC without launching the engine.

    XcomInterceptionEnv.__init__ only sets up the gym spaces; it does not
    launch the engine process (that only happens inside reset()). So we can
    instantiate the env, hand it to PPO for space inspection, and never call
    reset() — no engine window opens.
    """
    env = XcomInterceptionEnv(
        headless    = True,
        socket_name = "openxcom_bridge_bc.sock",
        max_steps   = 1,
    )
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class  = XcomExtractor,
        features_extractor_kwargs = dict(features_dim=64),
        net_arch                  = dict(pi=[64, 64], vf=[64, 64]),
    )
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs = policy_kwargs,
        device        = device,
        verbose       = 0,
    )
    return model


# ── training loop ──────────────────────────────────────────────────────────────

def train_bc(
    demos_paths: List[str],
    out_path:    str,
    epochs:      int   = 20,
    batch_size:  int   = 128,
    lr:          float = 3e-4,
    device:      str   = "auto",
) -> None:
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    print(f"[bc_pretrain] device={device}", flush=True)
    print(f"[bc_pretrain] loading demos from {len(demos_paths)} file(s) …", flush=True)

    records = _load_datasets(demos_paths)
    if not records:
        raise ValueError(f"No records found in: {demos_paths}")

    actions_np = np.array([r["action"] for r in records], dtype=np.int64)
    obs_batch  = _build_obs_batch(records)

    action_counts = {int(a): int(n) for a, n in zip(*np.unique(actions_np, return_counts=True))}
    print(f"[bc_pretrain] {len(records)} steps  action_counts={action_counts}", flush=True)

    print("[bc_pretrain] building policy …", flush=True)
    model  = _make_policy(device)
    policy = model.policy
    policy.train()

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Convert actions to tensor for loss
    actions_th = th.tensor(actions_np, dtype=th.long, device=device)

    # Inverse-frequency class weights: rare critical actions (research confirm, etc.)
    # get higher weight so the model learns them despite low demo frequency.
    counts = th.zeros(N_ACTIONS, device=device)
    for a in actions_np:
        counts[a] += 1
    class_weights = (counts.sum() / (N_ACTIONS * counts.clamp(min=1))).float()

    # Convert obs arrays to tensors (SB3 obs_as_tensor handles Dict)
    obs_tensors = obs_as_tensor(obs_batch, device)

    N = len(records)
    indices = np.arange(N)

    t0 = time.time()
    print(f"[bc_pretrain] training for {epochs} epochs, batch={batch_size} …", flush=True)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            idx = indices[start : start + batch_size]
            idx_th = th.tensor(idx, dtype=th.long, device=device)

            # Slice obs batch
            obs_slice: Dict[str, th.Tensor] = {}
            for k, v in obs_tensors.items():
                obs_slice[k] = v[idx_th]

            acts_slice = actions_th[idx_th]

            # Policy forward: weighted cross-entropy (rare actions get higher weight)
            dist = policy.get_distribution(obs_slice)
            loss = th.nn.functional.cross_entropy(
                dist.distribution.logits, acts_slice, weight=class_weights
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Accuracy metric
        with th.no_grad():
            dist_eval  = policy.get_distribution(obs_tensors)
            pred_actions = dist_eval.distribution.probs.argmax(dim=-1)
            accuracy = (pred_actions == actions_th).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"acc={accuracy:.3f}", flush=True)

    dt = time.time() - t0
    print(f"[bc_pretrain] training done in {dt:.1f}s", flush=True)

    # Final accuracy per action
    with th.no_grad():
        dist_final = policy.get_distribution(obs_tensors)
        preds = dist_final.distribution.probs.argmax(dim=-1).cpu().numpy()
    for a in sorted(action_counts):
        mask = actions_np == a
        acc_a = float((preds[mask] == a).mean()) if mask.any() else 0.0
        print(f"  action {a}: {acc_a:.3f} accuracy ({mask.sum()} samples)", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    model.save(out_path)
    print(f"[bc_pretrain] saved to {out_path}.zip", flush=True)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-cloning pretraining")
    p.add_argument("--demos", nargs="+",
                   default=[str(_ROOT / "python" / "data" / "demos.jsonl")],
                   help="One or more JSONL demo files (shell globs are expanded by the shell)")
    p.add_argument("--out",     default=os.path.join(CKPT_DIR, "bc_init"))
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--batch",   type=int,   default=128)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--device",  default="auto")
    args = p.parse_args()

    train_bc(
        demos_paths = args.demos,
        out_path    = args.out,
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
        device      = args.device,
    )


if __name__ == "__main__":
    main()
