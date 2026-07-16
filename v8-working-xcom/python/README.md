# xcom_bridge Python client

Connects to the Unix-domain socket bridge embedded in the OpenXcom engine,
receives live Geoscape telemetry (starting with `funds` every 5-second tick),
and logs every message to a replay file.

## Requirements

- Python 3.10+ on Windows 10 build 17063+
- No third-party packages (uses ctypes + stdlib only)

> **Why ctypes instead of `socket.AF_UNIX`?**  Python 3.9+ includes
> `socket.AF_UNIX` on Windows, but the support depends on internal CPython
> build details.  The ctypes path calls `ws2_32.dll` directly — identical to
> the working C test client — and has no version quirks.

## Running the client

```powershell
# From the repo root
cd python

# Stream to console and log to xcom_replay.jsonl (default)
python -m xcom_bridge.client

# Or specify a log path
python -m xcom_bridge.client my_run.jsonl
```

The engine must already be running and in the Geoscape for messages to flow.

## Launching the engine

### Normal launch (main menu, then New Game manually)

```powershell
.\engine\openxcom\build\bin\openxcom.exe
```

Then click **New Game → Beginner → OK**, wait for the "Choose base location"
globe, click a landmass, name your base, and let the Geoscape load.

### Auto-start (skips all menus, goes straight to Geoscape)

```powershell
$env:XCOM_BRIDGE_AUTOSTART = "1"
.\engine\openxcom\build\bin\openxcom.exe
```

The base is placed in central Europe (~11°E 48°N) with Beginner funding.
Unset the variable to return to normal:

```powershell
Remove-Item Env:XCOM_BRIDGE_AUTOSTART
```

## End-to-end test with fund change

1. Launch the engine (either mode above).
2. In a second terminal, start the client:
   ```powershell
   cd python
   python -m xcom_bridge.client
   ```
3. You should immediately see per-tick messages:
   ```
   Connecting to C:\Users\<you>\AppData\Local\Temp\openxcom_bridge.sock …
   Connected.  Logging to C:\…\python\xcom_replay.jsonl  (Ctrl-C to stop)

     funds: $    4,143,000
     funds: $    4,143,000
     …
   ```
4. **Spend money** — in the Geoscape, left-click your base icon → **Purchase /
   Recruit** → order any item.  The next tick will show a lower funds value.
5. Confirm the log file updated:
   ```powershell
   Get-Content python\xcom_replay.jsonl -Tail 5
   ```
   Expected:
   ```json
   {"ts": "2026-06-21T16:20:00.123456+00:00", "funds": 4143000}
   {"ts": "2026-06-21T16:20:05.234567+00:00", "funds": 4143000}
   {"ts": "2026-06-21T16:20:10.345678+00:00", "funds": 4112750}
   ```

## Log format

Each line is a self-contained JSON object:

```
{"ts": "<ISO-8601 UTC>", "funds": <int64>}
```

| Field   | Type   | Description                        |
|---------|--------|------------------------------------|
| `ts`    | string | UTC timestamp of receipt           |
| `funds` | int    | Current X-COM treasury (USD cents × 1000) |

### Why NDJSON and not Parquet?

The bridge is a real-time stream: each message must be persisted
immediately after receipt (a crash should lose at most one line).  Parquet
is a columnar batch format — appending a single row requires either
buffering many rows first (data-loss risk) or rewriting the file footer on
every append (very slow).  NDJSON gives crash-safe, dependency-free,
per-message durability with no trade-off for this workload.

When you have a complete run and want columnar storage, convert with two
lines:

```python
import pandas as pd
df = pd.read_json("xcom_replay.jsonl", lines=True)
df.to_parquet("xcom_replay.parquet")
```
