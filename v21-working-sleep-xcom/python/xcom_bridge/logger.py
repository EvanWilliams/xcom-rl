"""Replay logger: appends each bridge message as one JSON line to a .jsonl file.

Format choice — NDJSON (.jsonl) over Parquet:
  * Real-time append: each record is flushed immediately after receipt.
    Parquet is a columnar format designed for batch writes; appending a
    single row requires either buffering (risking data loss on crash) or
    repeatedly opening and rewriting the file footer.
  * Zero dependencies: NDJSON uses only stdlib json + io.  Parquet needs
    pyarrow (≥50 MB install).
  * Crash-safe: every line is a self-contained valid JSON object.  A killed
    process leaves a fully readable file up to the last complete line.
  * Interoperable: readable directly with jq, pandas.read_json(lines=True),
    or polars.read_ndjson().  Easy to batch-convert to Parquet later with
    two lines of pandas/polars if columnar storage becomes important.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

DEFAULT_LOG_PATH = Path("xcom_replay.jsonl")


class ReplayLogger:
    """Appends timestamped records to a newline-delimited JSON (NDJSON) file.

    Each line has the form (ts prepended to the engine message):
        {"ts": "<UTC ISO>", "type": "state", "event": "tick|purchase|build|sack|transfer",
         "game": {"funds": 4141000, "day": 1, "month": 1, "year": 1999,
                  "hour": 0, "minute": 0, "score": 0,
                  "countries": [{"id": "STR_USA", "funding": 600000, "satisfaction": 1}]},
         "bases": [{"name": "Bridge HQ", "lon": 0.2, "lat": 0.84,
                    "scientists": 0, "engineers": 0,
                    "facilities": [{"type": "STR_ACCESS_LIFT", "x": 2, "y": 2,
                                    "build_days": 0,
                                    "capacity": {"personnel":0,"storage":0,
                                                 "crafts":0,"labs":0,
                                                 "workshops":0,"psi_labs":0}}],
                    "stores": {"STR_PISTOL": 2},
                    "soldiers": [{"name": "John Smith", "rank": "STR_ROOKIE",
                                  "tu":50,"stamina":40,"health":30,"bravery":50,
                                  "reactions":45,"firing":55,"throwing":62,
                                  "strength":35,"psi_strength":0,"psi_skill":0,
                                  "melee":0,"armor":"STR_NONE_UC",
                                  "wound_recovery":0,"craft":null}],
                    "crafts": [{"type":"STR_SKYRANGER","status":"STR_READY",
                                "fuel":1500,"fuel_max":1500,"damage":0,
                                "damage_max":150,"soldiers":0,"weapons":[]}],
                    "transfers": []}]}
    """

    def __init__(self, path: Path = DEFAULT_LOG_PATH) -> None:
        self._path = path
        self._fh: IO[str] = open(path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: dict) -> None:
        """Write *record* plus a UTC timestamp as one JSON line."""
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()   # guarantees the line is on disk before the next recv

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ReplayLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
