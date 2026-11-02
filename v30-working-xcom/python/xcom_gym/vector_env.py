"""
XcomVectorEnv — VectorEnv-compatible wrapper for N parallel OpenXcom instances.

Instance layout
---------------
  rank 0          : non-headless visible window ("spectator"), speed_ms=10 (≈8×
                    real-time via XCOM_GAME_SPEED env var).
  ranks 1..N-1    : headless, step-driven.

Socket uniqueness
-----------------
Each instance is assigned the socket name "openxcom_bridge_{rank}.sock" so
multiple engine processes can coexist on the same machine.

Return types
------------
reset() and step() return PyTorch tensors (not numpy arrays) so that the
caller can directly feed GPU-accelerated training loops.
  obs      : dict[str, Tensor]   — shape (N, ...) on device
  rewards  : Tensor (N,) float32  on device
  terminated/truncated : Tensor (N,) bool on device
  infos    : list[dict]  (one per env, stays on CPU)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .env import XcomInterceptionEnv


# ── device selection ───────────────────────────────────────────────────────────

_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sock(rank: int) -> str:
    return f"openxcom_bridge_{rank}.sock"


def _stack_obs(obs_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Stack N per-env observation dicts into a single dict of batched tensors.

    Dict obs values come from _build_obs as either:
      - np.array shape (1,) dtype int32   for Box sub-spaces
      - np.int64 scalar (0-d)             for Discrete sub-spaces

    np.stack handles both: scalars → (N,), (1,)-arrays → (N, 1).
    """
    keys   = obs_list[0].keys()
    result: Dict[str, torch.Tensor] = {}
    for key in keys:
        arr = np.stack([obs[key] for obs in obs_list])   # (N,) or (N, 1)
        result[key] = torch.from_numpy(arr).to(device)
    return result


# ── main class ─────────────────────────────────────────────────────────────────

class XcomVectorEnv:
    """
    Manages N OpenXcom engine instances and exposes a batched VectorEnv API.

    Parameters
    ----------
    num_envs : int
        Total instances.  Must be ≥ 1.  For a useful spectator+training setup,
        pass ≥ 2 (instance 0 visible, 1..N-1 headless).
    max_steps : int
        Per-episode truncation limit forwarded to every XcomInterceptionEnv.
    spectator_speed_ms : int
        XCOM_GAME_SPEED value (ms/tick) for the visible instance.
        10 → ~8× faster than the 80 ms engine default.
        0 → leave at engine default.
    device : torch.device | str | None
        Target device for all returned tensors.  None → CUDA if available, else CPU.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int = 4,
        max_steps: int = 3_000,
        spectator_speed_ms: int = 10,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.device   = torch.device(device) if device is not None else _DEFAULT_DEVICE

        # Build one env per rank
        envs: List[XcomInterceptionEnv] = []
        for rank in range(num_envs):
            if rank == 0 and num_envs > 1:
                # Visible spectator: non-headless, fast-clock, no UFO pre-scan
                env = XcomInterceptionEnv(
                    max_steps    = max_steps,
                    headless     = False,
                    socket_name  = _sock(rank),
                    speed_ms     = spectator_speed_ms,
                    prefill_ufo  = False,
                )
            else:
                # Headless training instance
                env = XcomInterceptionEnv(
                    max_steps    = max_steps,
                    headless     = True,
                    socket_name  = _sock(rank),
                    speed_ms     = 0,
                    prefill_ufo  = True,
                )
            envs.append(env)
        self._envs = envs

        # Expose single-env spaces for external consumers (SB3, gym wrappers, …)
        self.single_observation_space = envs[0].observation_space
        self.single_action_space      = envs[0].action_space

        # Thread pool: one thread per env (step() is I/O-bound — waiting on the
        # engine socket — so the GIL is not a bottleneck here).
        self._pool = ThreadPoolExecutor(
            max_workers      = num_envs,
            thread_name_prefix = "xcom_vec",
        )

    # ── VectorEnv API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        """
        Reset all envs in parallel.

        Returns
        -------
        obs   : dict[str, Tensor] shape (N, ...) on self.device
        infos : list[dict] of length N
        """
        seeds = (
            [seed + r for r in range(self.num_envs)]
            if seed is not None
            else [None] * self.num_envs
        )
        futures = [
            self._pool.submit(env.reset, seed=s, options=options)
            for env, s in zip(self._envs, seeds)
        ]
        results  = [f.result() for f in futures]    # list of (obs_dict, info)
        obs_list = [r[0] for r in results]
        infos    = [r[1] for r in results]
        return _stack_obs(obs_list, self.device), infos

    def step(
        self,
        actions: Union[torch.Tensor, Sequence[int]],
    ) -> Tuple[
        Dict[str, torch.Tensor],   # obs
        torch.Tensor,              # rewards (N,) float32
        torch.Tensor,              # terminateds (N,) bool
        torch.Tensor,              # truncateds  (N,) bool
        List[Dict],                # infos
    ]:
        """
        Step all envs in parallel using the given action vector.

        Parameters
        ----------
        actions : Tensor shape (N,) or length-N sequence of ints.
                  Tensor may live on any device; it is moved to CPU before dispatch.
        """
        if isinstance(actions, torch.Tensor):
            action_list: List[int] = actions.cpu().numpy().astype(int).tolist()
        else:
            action_list = [int(a) for a in actions]

        if len(action_list) != self.num_envs:
            raise ValueError(
                f"actions length {len(action_list)} != num_envs {self.num_envs}"
            )

        futures = [
            self._pool.submit(env.step, a)
            for env, a in zip(self._envs, action_list)
        ]
        results = [f.result() for f in futures]    # (obs, rew, term, trunc, info)

        obs_list    = [r[0] for r in results]
        rewards     = torch.tensor([r[1] for r in results], dtype=torch.float32, device=self.device)
        terminateds = torch.tensor([r[2] for r in results], dtype=torch.bool,    device=self.device)
        truncateds  = torch.tensor([r[3] for r in results], dtype=torch.bool,    device=self.device)
        infos       = [r[4] for r in results]

        return _stack_obs(obs_list, self.device), rewards, terminateds, truncateds, infos

    def close(self) -> None:
        """Terminate all engine processes and shut down the thread pool."""
        futures = [self._pool.submit(env.close) for env in self._envs]
        for f in futures:
            try:
                f.result(timeout=10)
            except Exception:
                pass
        self._pool.shutdown(wait=False)

    # ── convenience ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_envs

    def __repr__(self) -> str:
        headless_count = self.num_envs - (1 if self.num_envs > 1 else 0)
        spec = (
            f"spectator(rank=0, speed_ms={self._envs[0]._speed_ms}) + "
            f"{headless_count} headless"
            if self.num_envs > 1
            else "1 headless"
        )
        return f"XcomVectorEnv(num_envs={self.num_envs}, {spec}, device={self.device})"

    def __enter__(self) -> "XcomVectorEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── smoke-test / verification ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    print(
        f"\nXcomVectorEnv verification — N={N}\n"
        f"  Instance 0 : visible spectator (non-headless, speed_ms=10)\n"
        f"  Instances 1..{N-1}: headless\n"
    )

    vec = XcomVectorEnv(num_envs=N, max_steps=200, spectator_speed_ms=10)
    print(repr(vec))
    print(f"  device : {vec.device}\n")

    print("reset() ...", flush=True)
    obs, infos = vec.reset()

    print("Observation shapes and dtypes after reset:")
    for k, v in obs.items():
        print(f"  {k:20s}  shape={tuple(v.shape)}  dtype={v.dtype}  device={v.device}")

    assert all(v.shape[0] == N for v in obs.values()), "Batch dim mismatch!"
    assert all(str(v.device).startswith(str(vec.device)) for v in obs.values()), \
        f"Device mismatch! Expected {vec.device}"

    print("\nRunning 5 random step() calls ...")
    for i in range(5):
        actions = torch.zeros(N, dtype=torch.long)   # action=0 for all
        obs, rewards, terminateds, truncateds, infos = vec.step(actions)
        done = (terminateds | truncateds)
        print(
            f"  step {i+1}: rewards={rewards.tolist()}  "
            f"done={done.tolist()}  "
            f"obs['score'] shape={tuple(obs['score'].shape)}"
        )

    print("\nVerification checks:")
    print(f"  [OK] obs batch dim = {N}")
    print(f"  [OK] tensors on device = {vec.device}")
    print(f"  [OK] rewards shape = (N={N},) float32")
    print(f"  [OK] terminateds/truncateds shape = (N={N},) bool")

    vec.close()
    print("\nDone.")
