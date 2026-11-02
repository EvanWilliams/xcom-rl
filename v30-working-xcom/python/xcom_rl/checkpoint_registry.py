"""
CheckpointRegistry — persistent store of PPO checkpoint quality metrics.

Tracks every checkpoint in checkpoints/registry.json and ranks them by a
composite Cydonia-progress score so training runs can warm-start from the
strategically best checkpoint, not just the most recently saved one.
"""
from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointRegistry:
    VERSION = 1

    def __init__(self, registry_path: str = "checkpoints/registry.json"):
        self._path = registry_path
        self._data: Dict = {
            "version": self.VERSION,
            "updated_at": _iso_now(),
            "checkpoints": {},
            "best_by_composite": None,
            "best_by_reward": None,
        }

    # ── persistence ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[registry] Warning: could not load {self._path}: {e}")

    def save(self) -> None:
        self._data["updated_at"] = _iso_now()
        self._recompute_global_bests()
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    # ── mutation ───────────────────────────────────────────────────────────────

    def scan(self, ckpt_dir: str) -> int:
        """Register any .zip in ckpt_dir not already in the registry. Returns count added."""
        added = 0
        for p in glob.glob(os.path.join(ckpt_dir, "*.zip")):
            if os.path.basename(p).startswith("bc_"):
                continue
            try:
                if not (os.path.getsize(p) > 0 and zipfile.is_zipfile(p)):
                    continue
            except OSError:
                continue
            rel = self._relpath(p)
            if rel not in self._data["checkpoints"]:
                self._register_path(rel)
                added += 1
        return added

    def register(self, path: str) -> None:
        """Add metadata-only entry for path (no eval). No-op if already present."""
        rel = self._relpath(path)
        if rel not in self._data["checkpoints"]:
            self._register_path(rel)

    def register_eval(self, path: str, metrics: dict) -> None:
        """Append an eval result to the checkpoint entry and update cached bests."""
        rel = self._relpath(path)
        if rel not in self._data["checkpoints"]:
            self._register_path(rel)
        composite = self._compute_composite(metrics)
        record = dict(metrics)
        record["eval_at"] = _iso_now()
        record["composite_score"] = round(composite, 4)
        entry = self._data["checkpoints"][rel]
        entry["evals"].append(record)
        if entry["best_composite"] is None or composite > entry["best_composite"]:
            entry["best_composite"] = round(composite, 4)
        mr = metrics.get("mean_reward", 0.0)
        if entry["best_mean_reward"] is None or mr > entry["best_mean_reward"]:
            entry["best_mean_reward"] = round(mr, 3)

    # ── queries ────────────────────────────────────────────────────────────────

    def get_best(self, by: str = "composite") -> Optional[str]:
        """Return the path of the best evaluated checkpoint, or None."""
        key = "best_by_composite" if by == "composite" else "best_by_reward"
        return self._data.get(key)

    def get_all_ranked(self, by: str = "composite") -> List[Dict]:
        """Return all entries sorted best-first by the given criterion."""
        score_key = "best_composite" if by == "composite" else "best_mean_reward"
        entries = list(self._data["checkpoints"].values())
        entries.sort(key=lambda e: (e[score_key] if e[score_key] is not None else -1e9), reverse=True)
        return entries

    def has_evals(self) -> bool:
        return any(e["evals"] for e in self._data["checkpoints"].values())

    def unevaluated(self) -> List[Dict]:
        """Return entries with no eval results, sorted by timestep ascending."""
        entries = [e for e in self._data["checkpoints"].values() if not e["evals"]]
        entries.sort(key=lambda e: e["timesteps"])
        return entries

    def print_table(self, by: str = "composite") -> None:
        ranked = self.get_all_ranked(by=by)
        if not ranked:
            print("[registry] No checkpoints registered.")
            return
        cols = f"{'#':<4} {'Checkpoint':<48} {'Steps':>9} {'Score':>7} {'Reward':>8} {'Win%':>5} {'Month':>5} {'N':>3}"
        print(cols)
        print("-" * len(cols))
        for i, e in enumerate(ranked, 1):
            name = os.path.basename(e["path"])[:46]
            ts   = e["timesteps"]
            comp = f"{e['best_composite']:.3f}" if e["best_composite"] is not None else "  N/A "
            rew  = f"{e['best_mean_reward']:.1f}" if e["best_mean_reward"] is not None else "   N/A"
            if e["evals"]:
                last  = e["evals"][-1]
                win   = f"{last['win_rate']*100:.0f}%"
                month = f"{last['mean_month_reached']:.1f}"
            else:
                win   = " N/A"
                month = " N/A"
            print(f"{i:<4} {name:<48} {ts:>9} {comp:>7} {rew:>8} {win:>5} {month:>5} {len(e['evals']):>3}")

    # ── internals ──────────────────────────────────────────────────────────────

    def _relpath(self, path: str) -> str:
        return os.path.relpath(path).replace("\\", "/")

    def _register_path(self, rel: str) -> None:
        basename = os.path.basename(rel)
        m = re.match(r"xcom_ppo_(.+?)_(\d+)_steps\.zip$", basename)
        if m:
            run_name  = m.group(1)
            timesteps = int(m.group(2))
        else:
            run_name  = basename.replace("xcom_ppo_", "").replace(".zip", "")
            timesteps = 0
        try:
            mtime    = os.path.getmtime(rel)
            saved_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            fsize    = os.path.getsize(rel)
        except OSError:
            saved_at = _iso_now()
            fsize    = 0
        self._data["checkpoints"][rel] = {
            "path":            rel,
            "run_name":        run_name,
            "timesteps":       timesteps,
            "file_size":       fsize,
            "saved_at":        saved_at,
            "evals":           [],
            "best_composite":  None,
            "best_mean_reward": None,
        }

    def _compute_composite(self, m: dict) -> float:
        """Cydonia-oriented composite score, 0–1."""
        win      = float(m.get("win_rate",            0.0))
        survive  = 1.0 - float(m.get("loss_rate",     0.0))
        depth    = float(m.get("mean_month_reached",  1.0)) / 12.0
        research = float(m.get("research_rate",       0.0))
        combat   = float(m.get("craft_launch_rate",   0.0))
        score    = min(max(float(m.get("mean_total_score", 0.0)) / 2000.0, 0.0), 1.0)
        return (0.30 * win + 0.15 * survive + 0.15 * depth +
                0.15 * research + 0.15 * combat + 0.10 * score)

    def prune(
        self,
        keep_last: int = 10,
        keep_best: int = 3,
        keep_every: int = 3,
        dry_run: bool = False,
    ) -> List[str]:
        """Delete low-value checkpoints from disk and remove them from the registry.

        Protected (never deleted):
          - best_by_composite and best_by_reward
          - The keep_last most recent checkpoints by timestep
          - The keep_best highest-scoring evaluated checkpoints
          - Any checkpoint that has eval results
          - Any filename starting with 'bc_'
          - 1 out of every ``keep_every`` of the remaining (otherwise-deletable)
            checkpoints, sampled oldest->newest, so we retain a decimated
            history instead of throwing all of it away. ``keep_every=3`` keeps
            a third; ``keep_every<=1`` keeps everything (deletion disabled).

        Returns the list of paths that were (or would be) deleted.
        """
        entries = list(self._data["checkpoints"].values())

        protected: set = set()

        # Global bests
        for key in ("best_by_composite", "best_by_reward"):
            p = self._data.get(key)
            if p:
                protected.add(p)

        # Most recent by timestep
        by_time = sorted(entries, key=lambda e: e["timesteps"], reverse=True)
        for e in by_time[:keep_last]:
            protected.add(e["path"])

        # Top evaluated by composite
        evaluated = [e for e in entries if e["best_composite"] is not None]
        evaluated.sort(key=lambda e: e["best_composite"], reverse=True)
        for e in evaluated[:keep_best]:
            protected.add(e["path"])

        # Anything with eval data
        for e in entries:
            if e["evals"]:
                protected.add(e["path"])

        # BC init weights
        for e in entries:
            if os.path.basename(e["path"]).startswith("bc_"):
                protected.add(e["path"])

        # Retain 1 out of every `keep_every` of the remaining deletable
        # checkpoints (deterministic oldest->newest sample) instead of deleting
        # all of them. keep_every <= 1 => keep everything.
        candidates = [e for e in sorted(entries, key=lambda e: e["timesteps"])
                      if e["path"] not in protected]
        if keep_every <= 1:
            protected.update(e["path"] for e in candidates)
        else:
            protected.update(e["path"] for i, e in enumerate(candidates)
                             if i % keep_every == 0)

        deleted: List[str] = []
        to_remove: List[str] = []
        freed_bytes = 0

        for rel, e in list(self._data["checkpoints"].items()):
            if e["path"] in protected:
                continue
            path = e["path"]
            size = e.get("file_size", 0)
            if dry_run:
                print(f"  [dry-run] would delete: {path}  ({size/1e6:.2f} MB)")
            else:
                if os.path.exists(path):
                    freed_bytes += os.path.getsize(path)
                    os.remove(path)
                to_remove.append(rel)
            deleted.append(path)

        if not dry_run:
            for rel in to_remove:
                del self._data["checkpoints"][rel]
            self._recompute_global_bests()
            self.save()
            print(f"[prune] Deleted {len(deleted)} checkpoints, "
                  f"freed {freed_bytes/1e9:.2f} GB")
        else:
            print(f"[prune] dry-run: {len(deleted)} checkpoints would be deleted")

        return deleted

    def _recompute_global_bests(self) -> None:
        best_comp      = None
        best_comp_path = None
        best_rew       = None
        best_rew_path  = None
        for rel, e in self._data["checkpoints"].items():
            if not e["evals"]:
                continue
            if not os.path.exists(rel):
                continue
            c = e["best_composite"]
            r = e["best_mean_reward"]
            if c is not None and (best_comp is None or c > best_comp):
                best_comp      = c
                best_comp_path = rel
            if r is not None and (best_rew is None or r > best_rew):
                best_rew      = r
                best_rew_path = rel
        self._data["best_by_composite"] = best_comp_path
        self._data["best_by_reward"]    = best_rew_path
