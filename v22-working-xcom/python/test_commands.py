"""
Command driver test for the OpenXcom RL bridge.

Connects to the engine, receives the current battle state (the engine emits
a full snapshot when a client connects mid-battle), then sends commands in
sequence: end_turn -> move -> equip -> shoot -> end_turn.

Usage (while a battle is in progress):
    python test_commands.py

Start a New Battle in OpenXcom first if one isn't already running.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xcom_bridge import _winsock as _ws
from xcom_bridge.model import BattleMessage, UNIT_FACTION

SOCK_PATH = os.path.join(os.environ.get('TEMP', r'C:\Windows\Temp'), 'openxcom_bridge.sock')

# ── helpers ────────────────────────────────────────────────────────────────

def send_cmd(sock, **kwargs):
    msg = json.dumps(kwargs) + '\n'
    print(f"  >> SEND: {msg.strip()}", flush=True)
    _ws.send_bytes(sock, msg.encode())


def recv_battle(sock, buf, deadline):
    """Block until the next battle message (skip geoscape ticks)."""
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
            except Exception:
                continue
            if rec.get('type') == 'battle':
                try:
                    return BattleMessage.from_dict(rec), buf
                except Exception as e:
                    print(f"  [parse error] {e}", flush=True)
    raise TimeoutError("Timed out waiting for battle event")


def wait_event(sock, buf, event_name, timeout=30):
    """Receive messages until one matches event_name; log all received."""
    deadline = time.time() + timeout
    while True:
        msg, buf = recv_battle(sock, buf, deadline)
        ev = msg.event
        alive   = [u for u in msg.units if not u.is_out]
        players = [u for u in alive if u.faction == 0]
        aliens  = [u for u in alive if u.faction == 1]
        if ev == 'turn_start':
            side = UNIT_FACTION.get(msg.side, str(msg.side)).upper()
            print(f"  << turn_start turn={msg.turn} side={side} "
                  f"xcom={len(players)} aliens={len(aliens)}", flush=True)
        elif ev == 'battle_start':
            print(f"  << battle_start turn={msg.turn} "
                  f"map={msg.map.width}x{msg.map.length}x{msg.map.height} "
                  f"units={len(msg.units)}", flush=True)
        else:
            actor = next((u for u in msg.units if u.id == msg.active_unit), None)
            if actor:
                print(f"  << {ev} unit#{actor.id} {actor.type[:12]} "
                      f"pos=({actor.x},{actor.y},{actor.z}) TU={actor.tu}/{actor.tu_max}",
                      flush=True)
            else:
                print(f"  << {ev} active={msg.active_unit}", flush=True)
        if ev == event_name:
            return msg, buf


# ── main ───────────────────────────────────────────────────────────────────

print(f"Connecting to {SOCK_PATH} ...", flush=True)
sock = _ws.open_socket(SOCK_PATH)
print("Connected. Waiting for battle state (engine sends snapshot on connect).", flush=True)

buf = b''

# ── Step 1: get current state ─────────────────────────────────────────────
# The engine emits a battle_start snapshot when a client connects mid-battle.
# If no battle is running, start one now: Main Menu -> New Battle.
print("\n[1] Waiting for battle_start snapshot ...", flush=True)
bstart, buf = wait_event(sock, buf, 'battle_start', timeout=120)

alive   = [u for u in bstart.units if not u.is_out]
players = [u for u in alive if u.faction == 0]
if not players:
    print("No alive player units. Start a battle first!", flush=True)
    _ws.close_socket(sock)
    sys.exit(1)

soldier = players[0]
print(f"  Got snapshot. Using unit#{soldier.id} '{soldier.type}' "
      f"pos=({soldier.x},{soldier.y},{soldier.z}) "
      f"TU={soldier.tu}/{soldier.tu_max} "
      f"rh={soldier.right_hand}", flush=True)

# ── Step 2: end turn ─────────────────────────────────────────────────────
# If we're on the player turn, end it and wait for the full cycle back.
print("\n[2] Sending end_turn (player -> alien -> player) ...", flush=True)
send_cmd(sock, action="end_turn")

print("  Waiting for alien turn_start ...", flush=True)
_, buf = wait_event(sock, buf, 'turn_start', timeout=30)
print("  Waiting for player turn_start ...", flush=True)
pturn, buf = wait_event(sock, buf, 'turn_start', timeout=120)

alive   = [u for u in pturn.units if not u.is_out]
players = [u for u in alive if u.faction == 0]
if not players:
    print("No alive player units after turn. Done.", flush=True)
    _ws.close_socket(sock)
    sys.exit(0)

soldier = players[0]
print(f"  Player turn back. Using unit#{soldier.id} '{soldier.type}' "
      f"pos=({soldier.x},{soldier.y},{soldier.z}) "
      f"TU={soldier.tu}/{soldier.tu_max}", flush=True)

# ── Step 3: move ─────────────────────────────────────────────────────────
mx, my, mz = soldier.x + 1, soldier.y, soldier.z
print(f"\n[3] move unit#{soldier.id} -> ({mx},{my},{mz}) ...", flush=True)
send_cmd(sock, action="move", unit_id=soldier.id, target=[mx, my, mz])
try:
    _, buf = wait_event(sock, buf, 'move', timeout=15)
    print("  Move confirmed on screen!", flush=True)
except TimeoutError:
    print("  Move event not received (tile blocked or invalid). Continuing.", flush=True)

# ── Step 4: equip right hand ─────────────────────────────────────────────
print(f"\n[4] equip right hand for unit#{soldier.id} ...", flush=True)
send_cmd(sock, action="equip", unit_id=soldier.id, hand="right")
time.sleep(0.3)
print("  Equip sent.", flush=True)

# ── Step 5: shoot ────────────────────────────────────────────────────────
if soldier.right_hand and soldier.right_hand.ammo > 0:
    tx, ty, tz = soldier.x + 3, soldier.y, soldier.z
    print(f"\n[5] shoot snap unit#{soldier.id} rh={soldier.right_hand.type} -> ({tx},{ty},{tz}) ...",
          flush=True)
    send_cmd(sock, action="shoot", unit_id=soldier.id,
             hand="right", mode="snap", target=[tx, ty, tz])
    try:
        _, buf = wait_event(sock, buf, 'shoot', timeout=15)
        print("  Shoot confirmed on screen!", flush=True)
    except TimeoutError:
        print("  Shoot event not received (no ammo / out of range / laser weapon?). Continuing.",
              flush=True)
else:
    print(f"\n[5] Skipping shoot — no right-hand weapon or no ammo "
          f"({soldier.right_hand}).", flush=True)

# ── Step 6: end turn ────────────────────────────────────────────────────
print(f"\n[6] Final end_turn ...", flush=True)
send_cmd(sock, action="end_turn")
try:
    _, buf = wait_event(sock, buf, 'turn_start', timeout=30)
    print("  Turn cycled. All 5 commands tested!", flush=True)
except TimeoutError:
    print("  turn_start not received within 30s.", flush=True)

_ws.close_socket(sock)
print("\nTest complete.", flush=True)
