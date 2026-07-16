"""
Geoscape command test for the OpenXcom RL bridge.

Tests each of the 5 geoscape commands in turn:
  1. research_assign  — start Laser Pistol research with 3 scientists
  2. manufacture_assign — start Laser Pistol manufacture with 2 engineers (qty 5)
  3. purchase         — buy 3 Rifles
  4. build_facility   — queue a General Stores at (3,3)
  5. launch_craft     — send first Skyranger toward first detected UFO

Usage:
    python test_geo_commands.py

Start OpenXcom and load a save in the Geoscape before running.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xcom_bridge import _winsock as _ws

SOCK_PATH = os.path.join(os.environ.get('TEMP', r'C:\Windows\Temp'), 'openxcom_bridge.sock')

# ── helpers ────────────────────────────────────────────────────────────────

def send_cmd(sock, **kwargs):
    msg = json.dumps(kwargs) + '\n'
    print(f"  >> SEND: {msg.strip()}", flush=True)
    _ws.send_bytes(sock, msg.encode())


def recv_next(sock, buf, timeout_ms=8000):
    """Block until the next JSON message arrives."""
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        if not _ws.recv_ready(sock, timeout_ms=200):
            continue
        chunk = _ws.recv_bytes(sock, 8192)
        if chunk is None:
            raise ConnectionError("Server closed connection")
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
                return rec, buf
            except Exception:
                continue
    raise TimeoutError("Timed out waiting for a message")


def drain_until_tick(sock, buf, max_ticks=5):
    """Discard messages until we see a 'tick' geoscape state, confirming the
    engine processed our command.  Returns the tick message and remaining buf."""
    for _ in range(max_ticks * 20):   # generous upper bound
        try:
            rec, buf = recv_next(sock, buf, timeout_ms=2000)
        except TimeoutError:
            break
        if rec.get('type') == 'state':
            return rec, buf
    return None, buf


# ── connect ────────────────────────────────────────────────────────────────

print(f"Connecting to {SOCK_PATH} ...", flush=True)
sock = _ws.open_socket(SOCK_PATH)
print("Connected.", flush=True)
buf = b''

# Drain any pending messages (old ticks, etc.)
print("Draining initial messages ...", flush=True)
t0 = time.time()
while time.time() - t0 < 2.0:
    try:
        _, buf = recv_next(sock, buf, timeout_ms=400)
    except TimeoutError:
        break

print("Ready. Starting tests.\n", flush=True)

# ── 1. research_assign ─────────────────────────────────────────────────────
print("[1] research_assign: start Laser Pistol with 3 scientists at base 0", flush=True)
send_cmd(sock, action="research_assign", base=0, project="STR_LASER_PISTOL", scientists=3)
tick, buf = drain_until_tick(sock, buf)
if tick:
    bases = tick.get('bases', [])
    if bases:
        print(f"  OK — base[0] scientists free: {bases[0].get('scientists', '?')}", flush=True)
    else:
        print("  OK — tick received (no base detail in state)", flush=True)
else:
    print("  WARNING: no tick received after research_assign", flush=True)

# ── 2. manufacture_assign ──────────────────────────────────────────────────
print("\n[2] manufacture_assign: produce 5 Laser Pistols with 2 engineers at base 0", flush=True)
send_cmd(sock, action="manufacture_assign", base=0, project="STR_LASER_PISTOL", engineers=2, amount=5)
tick, buf = drain_until_tick(sock, buf)
if tick:
    print("  OK — tick received", flush=True)
else:
    print("  WARNING: no tick received after manufacture_assign", flush=True)

# ── 3. purchase ────────────────────────────────────────────────────────────
print("\n[3] purchase: buy 3 Rifles", flush=True)
send_cmd(sock, action="purchase", base=0, type="item", id="STR_RIFLE", qty=3)
tick, buf = drain_until_tick(sock, buf)
if tick:
    funds = tick.get('funds', '?')
    print(f"  OK — funds now: {funds}", flush=True)
else:
    print("  WARNING: no tick received after purchase", flush=True)

# ── 4. build_facility ──────────────────────────────────────────────────────
print("\n[4] build_facility: queue General Stores at (3,3) in base 0", flush=True)
send_cmd(sock, action="build_facility", base=0, facility="STR_GENERAL_STORES", x=3, y=3)
tick, buf = drain_until_tick(sock, buf)
if tick:
    funds = tick.get('funds', '?')
    print(f"  OK — funds now: {funds}", flush=True)
else:
    print("  WARNING: no tick received after build_facility", flush=True)

# ── 5. launch_craft ────────────────────────────────────────────────────────
print("\n[5] launch_craft: send Skyranger-1 to first UFO (index 0)", flush=True)
send_cmd(sock, action="launch_craft", base=0, craft="Skyranger-1",
         target_type="ufo", target_id=0)
tick, buf = drain_until_tick(sock, buf)
if tick:
    print("  OK — tick received. Check Geoscape: craft should be en-route.", flush=True)
else:
    print("  WARNING: no tick received (maybe no UFO detected yet)", flush=True)

# ── done ───────────────────────────────────────────────────────────────────
_ws.close_socket(sock)
print("\nAll 5 geoscape commands sent. Verify visually in the Geoscape.", flush=True)
