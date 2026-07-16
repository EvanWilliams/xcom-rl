"""OpenXcom RL bridge client.

Connects to the Unix-domain socket exposed by the engine bridge, receives
newline-delimited JSON state messages, parses them into typed dataclasses,
prints a compact summary per tick and an expanded view on roster-change
events (purchase/sack/build/transfer), and appends every record with a UTC
timestamp to a NDJSON replay log.

Usage:
    python -m xcom_bridge.client              # log to xcom_replay.jsonl
    python -m xcom_bridge.client run.jsonl    # log to run.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from . import _winsock as _ws
from .logger import DEFAULT_LOG_PATH, ReplayLogger
from .model import Base, BattleMessage, BattleUnit, StateMessage, Transfer, UNIT_FACTION

SOCKET_PATH: str = os.path.join(
    os.environ.get("TEMP", r"C:\Windows\Temp"),
    os.environ.get("XCOM_BRIDGE_SOCKET", "openxcom_bridge.sock"),
)

# ── display helpers ────────────────────────────────────────────────────────

def _fmt_funds(n: int) -> str:
    return f"${n:,}"


def _base_summary(b: Base) -> str:
    under_construction = sum(1 for f in b.facilities if not f.is_built)
    pending = len(b.transfers)
    return (
        f"  [{b.name}] "
        f"{len(b.soldiers)} soldiers  "
        f"{len(b.crafts)} crafts  "
        f"{len(b.facilities)} facilities"
        + (f" ({under_construction} building)" if under_construction else "")
        + (f"  {pending} pending transfer(s)" if pending else "")
    )


def _print_tick(msg: StateMessage) -> None:
    g = msg.game
    print(
        f"[{g.date}] {_fmt_funds(g.funds):>14}  score={g.score:+}",
        flush=True,
    )
    for b in msg.bases:
        print(_base_summary(b), flush=True)


def _print_battle(msg: BattleMessage) -> None:
    alive   = [u for u in msg.units if not u.is_out]
    players = [u for u in alive if u.faction == 0]
    aliens  = [u for u in alive if u.faction == 1]

    if msg.event == "battle_start":
        print(
            f"\n{'='*60}\n"
            f"  BATTLE START  turn={msg.turn}  "
            f"map={msg.map.width}×{msg.map.length}×{msg.map.height}\n"
            f"  {len(players)} soldiers  {len(aliens)} aliens  "
            f"({len(msg.units)} total units)\n"
            f"{'='*60}",
            flush=True,
        )
    elif msg.event == "turn_start":
        side = UNIT_FACTION.get(msg.side, str(msg.side)).upper()
        print(
            f"\n--- Turn {msg.turn}  side={side}  "
            f"xcom={len(players)} aliens={len(aliens)} ---",
            flush=True,
        )
    else:
        # action event: find the active unit
        actor = next((u for u in msg.units if u.id == msg.active_unit), None)
        if actor:
            rh = f"{actor.right_hand.type}({actor.right_hand.ammo})" if actor.right_hand else "—"
            print(
                f"  [{msg.event:6s}] unit#{actor.id} {actor.type[:12]:12s} "
                f"({actor.faction_name[0].upper()})  "
                f"pos=({actor.x},{actor.y},{actor.z})  "
                f"TU={actor.tu}/{actor.tu_max}  HP={actor.health}/{actor.health_max}  "
                f"rh={rh}",
                flush=True,
            )
        else:
            print(f"  [{msg.event}] active_unit={msg.active_unit}", flush=True)


def _describe_transfer(t: Transfer) -> str:
    if t.kind == "soldier" and t.soldier:
        return f"soldier '{t.soldier.name}' ({t.soldier.rank}) in {t.hours}h"
    if t.kind == "craft" and t.craft:
        return f"craft {t.craft.type} in {t.hours}h"
    if t.kind == "item" and t.item:
        return f"{t.qty}x {t.item} in {t.hours}h"
    if t.kind in ("scientists", "engineers"):
        return f"{t.qty} {t.kind} in {t.hours}h"
    return f"{t.kind} in {t.hours}h"


def _print_event(msg: StateMessage) -> None:
    g = msg.game
    evt = msg.event.upper()
    print(f"\n=== {evt} @ {g.date}  funds={_fmt_funds(g.funds)} ===", flush=True)
    for b in msg.bases:
        print(f"  Base: {b.name}", flush=True)
        if msg.event == "build":
            building = [f for f in b.facilities if not f.is_built]
            for fac in building:
                print(f"    + facility {fac.type} at ({fac.x},{fac.y})  {fac.build_days}d remaining", flush=True)
        if msg.event in ("purchase", "transfer"):
            for t in b.transfers:
                print(f"    >> in transit: {_describe_transfer(t)}", flush=True)
        if msg.event == "sack":
            print(f"    soldiers now: {[s.name for s in b.soldiers]}", flush=True)
        print(_base_summary(b), flush=True)
    print("", flush=True)


# ── main loop ─────────────────────────────────────────────────────────────

def run(log_path: Path = DEFAULT_LOG_PATH) -> None:
    print(f"Connecting to {SOCKET_PATH} …", flush=True)
    try:
        sock = _ws.open_socket(SOCKET_PATH)
    except (OSError, ConnectionRefusedError) as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected.  Logging to {log_path.resolve()}  (Ctrl-C to stop)\n", flush=True)

    with ReplayLogger(log_path) as log:
        buf = b""
        try:
            while True:
                if not _ws.recv_ready(sock, timeout_ms=200):
                    continue
                chunk = _ws.recv_bytes(sock)
                if chunk is None:
                    print("Server closed the connection.")
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    log.log(record)
                    msg_type = record.get("type")
                    if msg_type == "state":
                        try:
                            msg = StateMessage.from_dict(record)
                        except (KeyError, TypeError):
                            print(f"  [parse error] {text[:120]}", flush=True)
                            continue
                        if msg.event == "tick":
                            _print_tick(msg)
                        else:
                            _print_event(msg)
                    elif msg_type == "battle":
                        try:
                            bmsg = BattleMessage.from_dict(record)
                        except (KeyError, TypeError) as e:
                            print(f"  [battle parse error] {e}  {text[:120]}", flush=True)
                            continue
                        _print_battle(bmsg)
                    else:
                        print(f"  {record}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            _ws.close_socket(sock)


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    run(log_path)


if __name__ == "__main__":
    main()
