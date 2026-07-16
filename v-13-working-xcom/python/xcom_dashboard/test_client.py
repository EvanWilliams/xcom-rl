"""WebSocket test client for the XCom dashboard.

Connects to ws://localhost:8000/ws and prints a formatted one-liner per
message, confirming that live bridge updates flow through the relay.

Usage
-----
    py python/xcom_dashboard/test_client.py          # default URL
    py python/xcom_dashboard/test_client.py ws://host:port/ws
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

try:
    import websockets
    import websockets.exceptions
except ImportError:
    print("websockets not found: py -m pip install websockets")
    sys.exit(1)

URL     = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws"
TIMEOUT = 120   # give up after this many seconds of silence


async def main() -> None:
    print(f"Connecting to {URL} …", flush=True)
    try:
        async with websockets.connect(URL, ping_interval=None) as ws:
            print("Connected.  Waiting for bridge messages (Ctrl-C to stop).\n", flush=True)
            msg_count   = 0
            state_count = 0
            battle_count = 0
            t0 = time.monotonic()

            async for raw in ws:
                elapsed = time.monotonic() - t0
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"  [bad JSON] {raw[:100]}", flush=True)
                    continue

                t = obj.get("type", "?")

                if t == "keepalive":
                    print(f"  {elapsed:7.1f}s  [keepalive]", flush=True)

                elif t == "state":
                    state_count += 1
                    ev = obj.get("event", "?")
                    g  = obj.get("game", {})
                    bases = obj.get("bases", [])
                    ufos  = [u for u in obj.get("ufos", []) if u.get("detected")]
                    sites = obj.get("mission_sites", [])
                    date  = (f"{g.get('year')}-"
                             f"{g.get('month', 0):02d}-"
                             f"{g.get('day', 0):02d}")
                    funds = g.get("funds", 0)
                    soldiers = sum(len(b.get("soldiers", [])) for b in bases)
                    print(
                        f"  {elapsed:7.1f}s  state/{ev:<18s}  {date}  "
                        f"${funds:>13,}  sol={soldiers}  "
                        f"ufos={len(ufos)}  sites={len(sites)}",
                        flush=True,
                    )

                elif t == "battle":
                    battle_count += 1
                    ev    = obj.get("event", "?")
                    turn  = obj.get("turn", "?")
                    side  = obj.get("side", "?")
                    units = obj.get("units", [])
                    alive = [u for u in units if u.get("status", 0) < 6]
                    xcom  = sum(1 for u in alive if u.get("faction") == 0)
                    alien = sum(1 for u in alive if u.get("faction") == 1)
                    print(
                        f"  {elapsed:7.1f}s  battle/{ev:<16s}  "
                        f"turn={turn}  side={side}  "
                        f"xcom={xcom}  alien={alien}",
                        flush=True,
                    )

                else:
                    print(f"  {elapsed:7.1f}s  [{t}] {raw[:120]}", flush=True)

                msg_count += 1

    except websockets.exceptions.ConnectionClosedError as exc:
        print(f"\nConnection closed: {exc}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - t0
        print(
            f"\n--- {msg_count} messages in {elapsed:.1f}s "
            f"({state_count} state, {battle_count} battle) ---",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
