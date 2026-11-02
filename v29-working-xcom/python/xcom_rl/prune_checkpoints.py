"""
Delete low-value checkpoints from python/checkpoints/ and update registry.json.

Protected checkpoints (never deleted):
  - best_by_composite and best_by_reward (highest quality evals)
  - The --keep-last most recent checkpoints by training step
  - The --keep-best highest-scoring evaluated checkpoints
  - Any checkpoint that has eval data
  - bc_*.zip (behaviour-cloning init weights)

Usage:
    py -3 -m xcom_rl.prune_checkpoints --dry-run
    py -3 -m xcom_rl.prune_checkpoints
    py -3 -m xcom_rl.prune_checkpoints --keep-last 5 --keep-best 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_rl.checkpoint_registry import CheckpointRegistry
from xcom_rl.train_ppo import CKPT_DIR


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prune old XCOM RL checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--keep-last", type=int, default=10,
                   help="Number of most-recent checkpoints to keep (default: 10)")
    p.add_argument("--keep-best", type=int, default=3,
                   help="Number of top-scoring evaluated checkpoints to keep (default: 3)")
    p.add_argument("--keep-every", type=int, default=3,
                   help="Retain 1 out of every N otherwise-deletable checkpoints "
                        "instead of deleting all of them (default: 3; use 1 to keep all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without touching anything")
    p.add_argument("--registry", type=str,
                   default=os.path.join(CKPT_DIR, "registry.json"),
                   help="Path to registry.json (default: checkpoints/registry.json)")
    args = p.parse_args()

    if not os.path.exists(CKPT_DIR):
        print(f"[prune] checkpoints directory not found: {CKPT_DIR}")
        sys.exit(1)

    registry = CheckpointRegistry(args.registry)
    registry.load()
    registry.scan(CKPT_DIR)

    total = len(registry._data["checkpoints"])
    print(f"[prune] Registry has {total} checkpoints.")
    print(f"[prune] keep_last={args.keep_last}  keep_best={args.keep_best}  "
          f"keep_every={args.keep_every}  dry_run={args.dry_run}")

    if args.dry_run:
        print()

    deleted = registry.prune(
        keep_last=args.keep_last,
        keep_best=args.keep_best,
        keep_every=args.keep_every,
        dry_run=args.dry_run,
    )

    remaining = total - len(deleted)
    print(f"[prune] {remaining} checkpoints remain protected.")

    if not args.dry_run:
        best_c = registry._data.get("best_by_composite")
        best_r = registry._data.get("best_by_reward")
        if best_c:
            print(f"[prune] best_by_composite: {os.path.basename(best_c)}")
        if best_r:
            print(f"[prune] best_by_reward:    {os.path.basename(best_r)}")


if __name__ == "__main__":
    main()
