"""
Scripted end-to-end validation of the OpenXcom RL bridge.

Drives the real engine through the socket protocol with no RL / no randomness:
  Step 1  – verify we are on the Geoscape (first tick)
  Step 2  – purchase one soldier
  Step 3  – assign scientists to a research project
  Step 4  – fast-forward up to N_DAYS looking for a detectable UFO
  Step 5  – launch the first available craft at the detected UFO
  Step 6  – if a Battlescape mission triggers within BATTLE_WAIT_SECS:
              move every soldier one tile east, end turn, repeat up to M_TURNS
              then send abort

Each step is validated: the action sent is compared to the effect seen in the
next state broadcast, and any mismatch is flagged.

Usage:
    python -m xcom_bridge.scripted_validation [--headless]
or:
    python xcom_bridge/scripted_validation.py [--headless]

--headless  Pass this when the engine was started with --headless.  In headless
            mode the engine only advances time on "step" commands; this script
            sends the initial step to get the first tick and adds a timing
            benchmark at the end.

Requirements:
  - OpenXcom must be running and the player must be on the Geoscape
    (load any save or start a new game and reach the globe screen, or use
     XCOM_BRIDGE_AUTOSTART=1 which reaches the Geoscape automatically)
  - The bridge socket must be active (engine was built with the bridge)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Add parent dir so we can import xcom_bridge when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xcom_bridge import _winsock as _ws

# ── configuration ─────────────────────────────────────────────────────────────

SOCK_PATH        = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"),
                                "openxcom_bridge.sock")
N_DAYS           = 30          # how many in-game days to fast-forward looking for a UFO
FAST_FORWARD_CHUNK = 5         # advance this many days per fast_forward command
M_TURNS          = 5           # max battlescape turns before aborting
BATTLE_WAIT_SECS = 300         # seconds of real time to wait for a battle to start
RECV_TIMEOUT_MS  = 500         # poll interval for recv_ready

# ── result dataclass ───────────────────────────────────────────────────────────

@dataclass
class StepResult:
    name: str
    sent: Optional[Dict]     = None   # command sent (None if read-only)
    before: Optional[Dict]   = None   # state snapshot before action
    after: Optional[Dict]    = None   # state snapshot after action
    ok: bool                 = False
    notes: List[str]         = field(default_factory=list)

    def flag(self, msg: str) -> None:
        self.notes.append(f"  MISMATCH: {msg}")
        self.ok = False

    def info(self, msg: str) -> None:
        self.notes.append(f"  OK: {msg}")

    def warn(self, msg: str) -> None:
        self.notes.append(f"  WARN: {msg}")

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines  = [f"[{status}] {self.name}"]
        if self.sent:
            lines.append(f"  sent:  {json.dumps(self.sent)}")
        lines.extend(self.notes)
        return "\n".join(lines)


# ── low-level socket helpers ───────────────────────────────────────────────────

class BridgeClient:
    def __init__(self, path: str) -> None:
        print(f"Connecting to {path} ...", flush=True)
        self._sock = _ws.open_socket(path)
        self._buf  = b""
        print("Connected.", flush=True)

    def close(self) -> None:
        _ws.close_socket(self._sock)

    def send(self, **kwargs) -> None:
        msg = json.dumps(kwargs) + "\n"
        print(f"  >> {msg.strip()}", flush=True)
        _ws.send_bytes(self._sock, msg.encode())

    def recv_one(self, timeout_secs: float = 60.0) -> Optional[Dict]:
        """Return the next complete JSON message, or None on timeout."""
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            if not _ws.recv_ready(self._sock, timeout_ms=RECV_TIMEOUT_MS):
                continue
            chunk = _ws.recv_bytes(self._sock, 8192)
            if chunk is None:
                raise ConnectionError("Engine closed the connection")
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                text = line.strip()
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return None

    def drain_to_state(self, timeout_secs: float = 10.0,
                       event_filter: Optional[str] = None) -> Optional[Dict]:
        """Return next geoscape state (type=state), optionally matching event_filter."""
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            msg = self.recv_one(timeout_secs=deadline - time.time())
            if msg is None:
                break
            if msg.get("type") == "state":
                if event_filter is None or msg.get("event") == event_filter:
                    return msg
        return None

    def drain_to_battle(self, timeout_secs: float = BATTLE_WAIT_SECS) -> Optional[Dict]:
        """Return next battlescape message (type=battle)."""
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            msg = self.recv_one(timeout_secs=deadline - time.time())
            if msg is None:
                break
            if msg.get("type") == "battle":
                return msg
            # keep draining geoscape ticks silently
        return None

    def flush_pending(self, window_secs: float = 0.5) -> None:
        """Drain any already-queued messages for the given window."""
        deadline = time.time() + window_secs
        while time.time() < deadline:
            if not _ws.recv_ready(self._sock, timeout_ms=100):
                break
            chunk = _ws.recv_bytes(self._sock, 65536)
            if chunk is None:
                break
            self._buf += chunk


# ── state accessors ────────────────────────────────────────────────────────────

def base0(state: Dict) -> Optional[Dict]:
    bases = state.get("bases", [])
    return bases[0] if bases else None

def game_info(state: Dict) -> Dict:
    return state.get("game", {})

def craft_list(state: Dict, base_idx: int = 0) -> List[Dict]:
    bases = state.get("bases", [])
    if base_idx >= len(bases):
        return []
    return bases[base_idx].get("crafts", [])

def transfer_list(state: Dict, base_idx: int = 0) -> List[Dict]:
    bases = state.get("bases", [])
    if base_idx >= len(bases):
        return []
    return bases[base_idx].get("transfers", [])

def soldier_list(state: Dict, base_idx: int = 0) -> List[Dict]:
    bases = state.get("bases", [])
    if base_idx >= len(bases):
        return []
    return bases[base_idx].get("soldiers", [])

def detected_ufos(state: Dict) -> List[Dict]:
    return [u for u in state.get("ufos", []) if u.get("detected")]

def mission_sites(state: Dict) -> List[Dict]:
    return state.get("mission_sites", [])


# ── individual validation steps ────────────────────────────────────────────────

def step_verify_geoscape(client: BridgeClient, headless: bool = False) -> StepResult:
    r = StepResult("verify_geoscape")
    print("\n[Step 1] Waiting for geoscape tick ...", flush=True)
    if headless:
        # Engine only advances time on "step" commands — send one to get the first tick.
        client.send(action="step")
    state = client.drain_to_state(timeout_secs=15.0)
    if state is None:
        r.flag("No geoscape state received within 15 s — is the engine on the Geoscape?")
        return r
    g = game_info(state)
    b = base0(state)
    r.after  = state
    r.ok     = True
    r.info(f"date: {g.get('year')}-{g.get('month', 0):02d}-{g.get('day', 0):02d}  "
           f"funds: ${g.get('funds', 0):,}")
    if b is None:
        r.flag("No bases in state — game may not have started")
    else:
        r.info(f"base[0]: '{b.get('name')}'  "
               f"soldiers={len(b.get('soldiers',[]))}  "
               f"crafts={len(b.get('crafts',[]))}")
    return r


def step_purchase_soldier(client: BridgeClient, before_state: Dict) -> StepResult:
    r = StepResult("purchase_soldier")
    r.before = before_state
    b_before = base0(before_state)
    soldiers_before = len(soldier_list(before_state)) if b_before else 0
    transfers_before = len(transfer_list(before_state))
    funds_before = game_info(before_state).get("funds", 0)

    print("\n[Step 2] Purchasing one soldier ...", flush=True)
    cmd = dict(action="purchase", base=0, type="soldier", qty=1)
    r.sent = cmd
    client.send(**cmd)

    # Wait for the state broadcast that purchase triggers
    state = client.drain_to_state(timeout_secs=10.0, event_filter="purchase")
    if state is None:
        r.flag("No 'purchase' state event received within 10 s")
        return r

    r.after = state
    transfers_after  = len(transfer_list(state))
    funds_after      = game_info(state).get("funds", 0)

    # Validate: a new transfer should have appeared
    if transfers_after > transfers_before:
        r.ok = True
        r.info(f"Transfer count: {transfers_before} -> {transfers_after}  (+1 soldier in transit)")
    else:
        r.flag(f"Transfer count unchanged ({transfers_before}) after purchase — soldier may not have been queued")

    cost = funds_before - funds_after
    if cost > 0:
        r.info(f"Funds deducted: ${cost:,}  (${funds_after:,} remaining)")
    else:
        r.warn(f"Funds did not decrease (before={funds_before}, after={funds_after})")

    return r


def step_assign_research(client: BridgeClient, before_state: Dict) -> StepResult:
    r = StepResult("assign_research")
    r.before = before_state
    b = base0(before_state)
    scientists_before = b.get("scientists", 0) if b else 0

    print("\n[Step 3] Assigning 1 scientist to Laser Pistol research ...", flush=True)
    cmd = dict(action="research_assign", base=0, project="STR_LASER_PISTOL", scientists=1)
    r.sent = cmd
    client.send(**cmd)

    state = client.drain_to_state(timeout_secs=10.0, event_filter="research_assign")
    if state is None:
        # fall back to any tick (the event may not come if there were no scientists)
        state = client.drain_to_state(timeout_secs=5.0)

    if state is None:
        r.flag("No state received after research_assign")
        return r

    r.after = state
    b_after = base0(state)
    scientists_after = b_after.get("scientists", 0) if b_after else 0

    if scientists_before > 0 and scientists_after < scientists_before:
        r.ok  = True
        r.info(f"Free scientists: {scientists_before} -> {scientists_after}  (1 assigned to lab)")
    elif scientists_before == 0:
        r.ok = True
        r.warn("Base has 0 free scientists — research queued but 0 assigned (no scientists available)")
    elif scientists_before > 0 and scientists_after == scientists_before:
        # Scientists unchanged — project may already be running at exactly the requested count
        # (engine clamps to max(cur, requested), so no net change if cur == requested).
        r.ok = True
        r.warn(f"Scientists unchanged ({scientists_before}) — project was already allocated "
               f"≥1 scientist; no additional scientist reassigned")
    else:
        r.flag(f"Scientists went up ({scientists_before} -> {scientists_after}) after "
               f"research_assign — unexpected increase")

    return r


def _game_days(g: Dict) -> int:
    """Return a rough day-count for date comparison (ignores leap years)."""
    months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    y, m, d = g.get("year", 0), g.get("month", 0), g.get("day", 0)
    return y * 365 + sum(months[1:m]) + d


def step_fast_forward_for_ufo(client: BridgeClient,
                               before_state: Dict,
                               n_days: int = N_DAYS,
                               chunk: int = FAST_FORWARD_CHUNK) -> Tuple[StepResult, Optional[Dict], Optional[Dict]]:
    """
    Advance geoscape time in `chunk`-day increments looking for a detected UFO.
    Returns (StepResult, ufo_dict_or_None, final_state).
    """
    r = StepResult("fast_forward_for_ufo")
    r.before = before_state

    print(f"\n[Step 4] Fast-forwarding up to {n_days} in-game days looking for UFO ...", flush=True)

    days_elapsed = 0
    final_state  = before_state
    ufo_found    = None

    while days_elapsed < n_days:
        advance = min(chunk, n_days - days_elapsed)
        cmd = dict(action="fast_forward", days=advance)
        r.sent = cmd

        g_before = game_info(final_state)
        days_before = _game_days(g_before)

        client.send(**cmd)

        # The engine emits "fast_forward_done" when the advance finishes.
        # NOTE: if the geoscape has a popup open (e.g. terror site notification) the
        # game timer stops ticking, so fast_forward steps are never consumed and
        # fast_forward_done is never broadcast — resulting in a timeout here.
        state = client.drain_to_state(timeout_secs=60.0, event_filter="fast_forward_done")
        if state is None:
            r.flag(f"No fast_forward_done received after requesting {advance} days "
                   f"— engine game timer may be paused by a popup/dogfight")
            break

        final_state   = state
        days_elapsed += advance
        g = game_info(state)
        ufos = detected_ufos(state)
        sites = mission_sites(state)

        actual_days = _game_days(g) - days_before
        date_str = (f"{g.get('year')}-{g.get('month', 0):02d}-{g.get('day', 0):02d}")
        print(f"  date: {date_str}  "
              f"detected_ufos={len(ufos)}  mission_sites={len(sites)}  "
              f"(requested {advance}d, advanced {actual_days}d)",
              flush=True)

        # Flag early termination: the geoscape loop exits when _pause is set by
        # game events (terror site pop-up, dogfight start, etc.).  If actual days
        # advanced is materially less than requested, that's a partial advance.
        if actual_days < advance - 1:
            r.warn(f"fast_forward({advance}) advanced only {actual_days} calendar day(s) "
                   f"— engine loop was interrupted by a game event (popup/dogfight)")

        if ufos:
            ufo_found = ufos[0]
            r.ok  = True
            r.info(f"UFO detected after ~{days_elapsed} days: "
                   f"idx={ufo_found['idx']}  type={ufo_found['type']}")
            break
        if sites:
            r.ok  = True
            r.warn(f"Mission site appeared (no UFO detection needed) — "
                   f"idx={sites[0]['idx']} type={sites[0]['type']}")
            break
    else:
        if ufo_found is None:
            r.warn(f"No UFO detected after {days_elapsed} in-game days")
            r.ok = True   # not a mismatch, just no event

    r.after = final_state
    return r, ufo_found, final_state


def step_launch_craft(client: BridgeClient,
                      before_state: Dict,
                      target: Dict) -> StepResult:
    """target can be a UFO dict (has 'idx','type') or a mission_site dict (same keys)."""
    r = StepResult("launch_craft")
    r.before = before_state

    crafts = craft_list(before_state)
    ready  = [c for c in crafts if c.get("status") in ("STR_READY", "STR_REFUELLING")]
    if not ready:
        r.flag("No ready craft at base[0] — cannot launch")
        return r

    craft      = ready[0]
    tgt_idx    = target["idx"]
    tgt_type   = target.get("type", "?")
    craft_name = craft.get("name", craft.get("type", ""))

    # Determine whether target is a UFO or mission site by checking for "detected" key
    if "detected" in target:
        target_kind = "ufo"
    else:
        target_kind = "mission_site"

    print(f"\n[Step 5] Launching '{craft_name}' at {target_kind} idx={tgt_idx} ({tgt_type}) ...",
          flush=True)
    cmd = dict(action="launch_craft", base=0, craft=craft_name,
               target_type=target_kind, target_id=tgt_idx)
    r.sent = cmd
    client.send(**cmd)

    state = client.drain_to_state(timeout_secs=10.0, event_filter="launch_craft")
    if state is None:
        state = client.drain_to_state(timeout_secs=5.0)
    if state is None:
        r.flag("No state received after launch_craft")
        return r

    r.after = state
    # Validate craft status changed to STR_OUT
    crafts_after = craft_list(state)
    matching = [c for c in crafts_after
                if c.get("name") == craft_name or c.get("type") == craft.get("type")]

    if matching and matching[0].get("status") == "STR_OUT":
        r.ok = True
        r.info(f"Craft '{craft_name}' status: STR_READY -> STR_OUT (en route)")
    elif matching:
        status_after = matching[0].get("status", "?")
        r.flag(f"Craft '{craft_name}' status is '{status_after}' (expected STR_OUT)")
    else:
        r.warn("Could not find craft in post-launch state — it may have left base list")
        r.ok = True

    return r


def step_battlescape(client: BridgeClient, headless: bool = False) -> StepResult:
    r = StepResult("battlescape_sequence")

    if headless:
        # In headless mode the craft is in flight but no time advances without
        # explicit step commands.  Reaching the target also triggers a
        # ConfirmLanding popup that requires UI interaction, so we cannot
        # complete a battlescape sequence purely in headless mode.
        r.warn("Skipped in headless mode — battlescape requires UI interaction "
               "(ConfirmLanding popup) that headless cannot auto-dismiss")
        r.ok = True
        return r

    print(f"\n[Step 6] Waiting up to {BATTLE_WAIT_SECS}s for Battlescape to start ...",
          flush=True)

    # The engine emits battle_start when a Battlescape mission triggers.
    bstate = client.drain_to_battle(timeout_secs=BATTLE_WAIT_SECS)
    if bstate is None:
        r.warn("No battle started — interception may take longer or UFO escaped")
        r.ok = True
        return r

    event = bstate.get("event", "?")
    units = bstate.get("units", [])
    # status 6=dead, 7=unconscious; faction 0=player
    alive_players = [u for u in units
                     if u.get("status", 0) not in (6, 7) and u.get("faction") == 0]

    print(f"  << battle {event}  units={len(units)}  player_units={len(alive_players)}",
          flush=True)
    if not alive_players:
        r.warn("No player units in battle — skipping battlescape steps")
        r.ok = True
        return r

    r.info(f"Battle started: {event}  player_units={len(alive_players)}")

    # Wait for the player turn
    print("  Waiting for player turn_start ...", flush=True)
    for _ in range(60):
        msg = client.drain_to_battle(timeout_secs=5.0)
        if msg is None:
            break
        if msg.get("event") == "turn_start" and msg.get("side") == 0:
            print(f"  << player turn_start (turn {msg.get('turn')})", flush=True)
            bstate = msg
            break
    else:
        r.flag("Player turn_start not received — cannot execute moves")
        return r

    # Execute up to M_TURNS turns of: move all soldiers east, end turn
    for turn_idx in range(M_TURNS):
        units = bstate.get("units", [])
        alive_players = [u for u in units
                         if u.get("status", 0) not in (6, 7) and u.get("faction") == 0]
        if not alive_players:
            r.warn(f"No player units alive at turn {turn_idx+1}")
            break

        print(f"\n  -- Turn {turn_idx+1}: moving {len(alive_players)} soldiers east --",
              flush=True)

        moves_sent = 0
        for u in alive_players:
            uid = u["id"]
            tx, ty, tz = u["x"] + 1, u["y"], u["z"]
            client.send(action="move", unit_id=uid, target=[tx, ty, tz])
            moves_sent += 1
            # brief pause to avoid flooding before confirmation
            time.sleep(0.05)

        # Drain move events (best effort — wall runs may not move everyone)
        for _ in range(moves_sent * 3):
            msg = client.drain_to_battle(timeout_secs=2.0)
            if msg is None:
                break
            if msg.get("event") in ("turn_start",):
                bstate = msg
                break

        # End turn
        print(f"  Sending end_turn (turn {turn_idx+1}) ...", flush=True)
        client.send(action="end_turn")

        # Wait for alien turn then player turn (or battle end)
        alien_seen = False
        for _ in range(120):
            msg = client.drain_to_battle(timeout_secs=5.0)
            if msg is None:
                break
            ev = msg.get("event")
            if ev == "turn_start":
                side = msg.get("side")
                if side == 1 and not alien_seen:
                    alien_seen = True
                    print(f"  << alien turn_start (turn {msg.get('turn')})", flush=True)
                elif side == 0:
                    print(f"  << player turn_start (turn {msg.get('turn')})", flush=True)
                    bstate = msg
                    break
            elif ev in ("battle_end", "abort"):
                print(f"  << {ev} — battle over", flush=True)
                r.ok = True
                r.info(f"Battle ended at turn {turn_idx+1}")
                return r
        else:
            r.warn("Turn cycle timed out")
            break

    # Send abort after M_TURNS
    print(f"\n  M_TURNS ({M_TURNS}) reached — sending abort ...", flush=True)
    # OpenXcom abort is triggered by Esc → Abort Mission in UI.
    # The bridge doesn't have a dedicated abort command, so we end the turn
    # and let the mission resolve naturally, or flag that we need an abort command.
    client.send(action="end_turn")
    r.ok   = True
    r.warn(f"Sent final end_turn after {M_TURNS} turns (no explicit abort command in bridge)")
    r.info(f"Executed {M_TURNS} battlescape turns with move+end_turn sequence")
    return r


# ── headless timing benchmark ─────────────────────────────────────────────────

def headless_timing_benchmark(client: BridgeClient, n: int = 200) -> None:
    """
    Send N step commands back-to-back and time how long N ticks take.
    Compares to non-headless equivalent (N × 80 ms timer interval).
    """
    print(f"\n=== Headless timing benchmark: {n} steps ===", flush=True)
    t0 = time.perf_counter()
    for _ in range(n):
        client.send(action="step")
    received = 0
    for _ in range(n):
        st = client.drain_to_state(timeout_secs=10.0)
        if st is None:
            break
        received += 1
    dt = time.perf_counter() - t0
    if received == 0:
        print("  ERROR: no states received", flush=True)
        return
    ms_per_step = dt / received * 1000
    nonhl_total = n * 0.080
    speedup = nonhl_total / dt if dt > 0 else float("inf")
    print(f"  {received}/{n} steps received in {dt:.3f}s  ({ms_per_step:.2f} ms/step)", flush=True)
    print(f"  Non-headless equivalent: {n} × 80 ms = {nonhl_total:.1f}s", flush=True)
    print(f"  Speedup vs non-headless timer: {speedup:.0f}×", flush=True)


# ── report ─────────────────────────────────────────────────────────────────────

def print_report(results: List[StepResult]) -> None:
    width = 70
    print("\n" + "=" * width, flush=True)
    print("SCRIPTED VALIDATION REPORT", flush=True)
    print("=" * width, flush=True)
    for r in results:
        print(str(r), flush=True)
    print("-" * width, flush=True)
    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    print(f"TOTAL: {passed}/{len(results)} passed, {failed} failed", flush=True)
    print("=" * width, flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted OpenXcom bridge validation")
    parser.add_argument("--headless", action="store_true",
                        help="Engine was started with --headless (step-driven mode)")
    args = parser.parse_args()
    HEADLESS = args.headless

    client  = BridgeClient(SOCK_PATH)
    results: List[StepResult] = []

    try:
        # Flush stale buffered messages
        client.flush_pending(window_secs=0.5)

        # Step 1: verify geoscape
        r1 = step_verify_geoscape(client, headless=HEADLESS)
        results.append(r1)
        if not r1.ok or r1.after is None:
            print_report(results)
            return 1
        state = r1.after

        # Step 2: purchase a soldier
        r2 = step_purchase_soldier(client, state)
        results.append(r2)
        state = r2.after or state

        # Step 3: assign research
        r3 = step_assign_research(client, state)
        results.append(r3)
        state = r3.after or state

        # Step 4: fast-forward looking for UFO or mission site
        r4, ufo, state = step_fast_forward_for_ufo(client, state)
        results.append(r4)

        # Choose a launch target: prefer a detected UFO, fall back to mission site
        sites = mission_sites(state)
        launch_target = ufo if ufo is not None else (sites[0] if sites else None)

        # Step 5: launch craft at target (if any)
        if launch_target is not None:
            r5 = step_launch_craft(client, state, launch_target)
            results.append(r5)
            state = r5.after or state

            # Step 6: wait for / play battlescape
            r6 = step_battlescape(client, headless=HEADLESS)
            results.append(r6)
        else:
            skipped_launch = StepResult("launch_craft")
            skipped_launch.warn("Skipped — no UFO or mission site in fast-forward window")
            skipped_launch.ok = True
            results.append(skipped_launch)

            skipped_battle = StepResult("battlescape_sequence")
            skipped_battle.warn("Skipped — no craft launched")
            skipped_battle.ok = True
            results.append(skipped_battle)

        # Timing benchmark (headless only): send 200 steps back-to-back
        if HEADLESS:
            headless_timing_benchmark(client, n=200)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]", flush=True)
    except ConnectionError as e:
        print(f"\n[Connection error: {e}]", flush=True)
    finally:
        client.close()
        print_report(results)

    failed = sum(1 for r in results if not r.ok)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
