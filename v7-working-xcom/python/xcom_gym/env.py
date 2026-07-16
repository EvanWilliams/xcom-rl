"""
Gymnasium Env — click-only interface to OpenXcom.

All actions are simulated mouse clicks (or Escape) injected via SDL, exactly
as a human player would interact with the game.  No internal API shortcuts.

Action space: Discrete(23)  — see ACTIONS table below.
Observation space: Dict — game state + screen_hint so the agent can tell
                          which dialog/screen is currently showing.

Reward:
  +0.002        every mouse click (named button or random_click)
  +0.000        wait / dismiss (no click)
  +10           UFO first detected on radar
  +50           craft dispatched (STR_OUT transition)
  +1000         game won (Cydonia)
  delta*0.001   routine strategic score ticks (|delta| < 10)
  delta*1.0     post-battle total_score delta (|delta| >= 10) — covers:
                  alien kills (Sectoid Soldier +10 … Ethereal Commander +40)
                  alien stuns (2× kill value)
                  civilian killed by X-COM  −50 | civilian killed by aliens −30
                  soldier killed/MIA −20 | HWP destroyed −20
                  civilian saved +30 | alien base destroyed +500
  delta*1.0     research score delta (score field, any positive delta) — covers:
                  Plasma Pistol Clip +5 … Cydonia or Bust +60 (§4.8 table)

The random_click action samples a uniform random (x, y) from the full
320×200 game screen at runtime so the agent can discover UI elements
not covered by named buttons.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xcom_bridge import _winsock as _ws

# ── paths ──────────────────────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_EXE = os.path.join(_ROOT, "engine", "openxcom", "build", "bin", "openxcom.exe")
ENGINE_CWD = os.path.join(_ROOT, "engine", "openxcom", "build", "bin")
_TEMP      = os.environ.get("TEMP", r"C:\Windows\Temp")
_LOG_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ── action table ───────────────────────────────────────────────────────────────
# Each entry: (name, x_or_None, y_or_None)
# x=None  →  special action (wait / dismiss)
# x≥0     →  click(x, y) injected as SDL mouse event
#
# Geoscape button positions (320×200 screen, sidebar at x=257..320):
#   screenWidth=320, screenHeight=200, screenHeight/2=100
#   _btnIntercept  TextButton(63,11, 257,  0) → centre (288,  5)
#   _btnBases      TextButton(63,11, 257, 12) → centre (288, 17)
#   _btnOptions    TextButton(63,11, 257, 48) → centre (288, 53)
#   _btn5Secs      TextButton(31,13, 257,112) → centre (272,118)
#   _btn1Min       TextButton(31,13, 289,112) → centre (304,118)
#   _btn5Mins      TextButton(31,13, 257,126) → centre (272,132)
#   _btn30Mins     TextButton(31,13, 289,126) → centre (304,132)
#   _btn1Hour      TextButton(31,13, 257,140) → centre (272,146)
#   _btn1Day       TextButton(31,13, 289,140) → centre (304,146)
#
# Dialog button positions (from source + previous analysis):
#   UfoDetectedState    Intercept(128,124) Centre(128,140) Cancel(128,156)
#   MissionDetectedState Intercept(128,138) Centre(128,158) Cancel(128,178)
#   InterceptState      row-1 craft (152,82)  Cancel (244,152)
#   ConfirmDestState    OK (93,110)  Cancel (163,110)
#   Dismiss (Escape)    → closes LowFuel/CraftError/MonthlyReport/ItemsArriving etc.

ACTIONS: list[tuple[str, Optional[int], Optional[int], int]] = [
    # (name, x, y, button)  button: 1=left  3=right
    # ── Geoscape idle / wait ────────────────────────────────────────────
    ("wait",                 None, None, 1),   # 0 — no-op; in headless sends step

    # ── Geoscape sidebar ────────────────────────────────────────────────
    ("geo_intercept",         288,    5, 1),   # 1 — open Intercept screen manually
    ("geo_bases",             288,   17, 1),   # 2 — go to Bases screen
    ("geo_options",           288,   53, 1),   # 3 — options / pause

    # ── Time speed buttons ──────────────────────────────────────────────
    ("speed_5sec",            272,  118, 1),   # 4
    ("speed_1min",            304,  118, 1),   # 5
    ("speed_5min",            272,  132, 1),   # 6
    ("speed_30min",           304,  132, 1),   # 7
    ("speed_1hour",           272,  146, 1),   # 8
    ("speed_1day",            304,  146, 1),   # 9

    # ── UFO Detected dialog ─────────────────────────────────────────────
    ("ufo_intercept",         128,  124, 1),   # 10
    ("ufo_centre",            128,  140, 1),   # 11
    ("ufo_cancel",            128,  156, 1),   # 12

    # ── Mission Detected dialog ─────────────────────────────────────────
    ("mission_intercept",     128,  138, 1),   # 13
    ("mission_cancel",        128,  178, 1),   # 14

    # ── Intercept screen ────────────────────────────────────────────────
    ("intercept_craft1",      152,   82, 1),   # 15 — first craft row
    ("intercept_cancel",      244,  152, 1),   # 16

    # ── Confirm Destination dialog ──────────────────────────────────────
    ("confirm_ok",             93,  110, 1),   # 17
    ("confirm_cancel",        163,  110, 1),   # 18

    # ── Dismiss (Escape) ────────────────────────────────────────────────
    ("dismiss",               None, None, 1),  # 19 — closes info dialogs, monthly report, etc.

    # ── Exploratory random clicks ─────────────────────────────────────────
    # x=-1 is a sentinel; _execute_action samples uniform random (x, y) at
    # runtime from the full 320×200 game screen.  Same click reward as named
    # buttons so the agent is motivated to probe undiscovered UI elements.
    ("random_click",            -1,   -1, 1),  # 20 — left-click at random position
    ("random_right_click",      -1,   -1, 3),  # 21 — right-click at random position

    # ── Keyboard escapes ─────────────────────────────────────────────────
    # Used to navigate screens that have no exposed click targets, e.g. the
    # "Create New Game" / name-entry screens that appear before the geoscape.
    # No click reward — keyboard actions earn 0 like wait/dismiss.
    ("key_a",                 None, None, 1),  # 22 — 'A' key
    ("key_enter",             None, None, 1),  # 23 — Enter / Return key

    # ── Main-menu escape ──────────────────────────────────────────────────
    # Sends the bridge "new_game" command which bypasses the GUI menu flow
    # and jumps directly to a fresh Beginner geoscape.  Only takes effect
    # when screen_type == main_menu; otherwise it is a no-op (like wait).
    # Earns MENU_ESCAPE_REWARD (+5) when the screen successfully transitions
    # from main_menu to geoscape.
    ("menu_new_game",         None, None, 1),  # 24 — bridge new_game (main-menu only)
]

N_ACTIONS    = len(ACTIONS)
ACTION_NAMES = [a[0] for a in ACTIONS]

# Sentinel value used in the ACTIONS table for random_click
_RANDOM_CLICK_SENTINEL = -1

# Full game-screen bounds (logical pixels, 320×200 native resolution)
SCREEN_W = 320
SCREEN_H = 200

# Screen hints → integer for obs encoding
SCREEN_TYPES: dict[str, int] = {
    "geoscape":         0,
    "ufo_detected":     1,
    "mission_detected": 2,
    "intercept":        3,
    "confirm_dest":     4,
    "low_fuel":         5,
    "craft_error":      6,
    "monthly_report":   7,
    "items_arriving":   8,
    "craft_patrol":     9,
    "other":            10,
    "main_menu":        11,   # engine at start/main-menu (no SavedGame yet)
}
N_SCREEN_TYPES = max(SCREEN_TYPES.values()) + 1

# ── Battlescape scoring reference (XCOM wiki §4.4) ────────────────────────────
# These are the game's internal point values that feed into total_score after a
# mission ends.  We don't parse them per-kill; instead we detect the post-battle
# score delta and apply BATTLE_SCORE_SCALE (see step() reward logic).
#
# Alien kill points (stun = 2× kill value):
#   Sectoid Soldier 10 | Sectoid Navigator 14 | Sectoid Medic 16
#   Sectoid Engineer 18 | Sectoid Leader 25 | Sectoid Commander 35
#   Floater Soldier 14 | Floater Navigator 15 | Floater Medic 16
#   Floater Engineer 18 | Floater Leader 25 | Floater Commander 35
#   Snakeman Soldier 15 | Snakeman Navigator 17 | Snakeman Engineer 20
#   Snakeman Leader 28 | Snakeman Commander 36
#   Ethereal Soldier 20 | Ethereal Leader 30 | Ethereal Commander 40
#   Muton Soldier 15 | Muton Navigator 17 | Muton Engineer 19
#   Reaper 25 | Chryssalid 25 | Silacoid 20 | Celatid 20
#   Sectopod 30 | Cyberdisc 20 | Zombie 18
#
# Mission outcome points:
#   Civilian killed by X-COM  -50 | Civilian killed by Aliens -30
#   X-COM Soldier MIA         -20 | X-COM Soldier Killed      -20*
#   HWP Destroyed             -20 | Civilian Saved            +30
#   Alien Base Destroyed      +500
#
# (* penalty scales up with soldier rank/experience)
#
# All of the above accumulate into total_score when the geoscape resumes.
# A delta >= BATTLE_SCORE_THRESHOLD on a single step is treated as a
# mission-end event and scaled by BATTLE_SCORE_SCALE instead of SCORE_SCALE.
BATTLE_SCORE_THRESHOLD = 10   # points; any |delta| >= this → battle scoring
BATTLE_SCORE_SCALE     = 1.0  # 1 game-point = 1 reward unit after a battle

# ── Research completion scoring reference (XCOM wiki §4.8) ───────────────────
# Points feed into save->getResearchScores(), which we broadcast as the "score"
# field.  Any positive delta in that field means a project just completed.
# Applied at RESEARCH_SCORE_SCALE = 1.0  (same philosophy as battle scoring).
#
#  5 pts : Plasma Pistol Clip
#  6 pts : Plasma Rifle Clip
#  7 pts : Heavy Plasma Clip
# 10 pts : Blaster Bomb | Craft Laser Cannon | Heavy Laser | Laser Pistol
#          Laser Rifle | Laser Weapons | Stun Bomb
# 15 pts : Laser Defenses | Tank/Laser Cannon
# 20 pts : Alien Grenade | Flying Suit | Medi-Kit | Motion Scanner
#          Personal Armor | Plasma Pistol | Power Suit | Psi-Amp
# 25 pts : Craft Fusion Weapon | Craft Plasma Weapon | Fusion Defenses
#          Grav Shield | Hovertank/Launcher | Hovertank/Plasma
#          Hyper-Wave Decoder | Mind Probe | Mind Shield | Plasma Defenses
#          Plasma Rifle | Psi-Lab
# 30 pts : Alien Abduction | Alien Alloys | Alien Entertainment | Alien Food
#          Alien Reproduction | Alien Surgery | Heavy Plasma
#          New Fighter Craft (Firestorm) | New Fighter-Transporter (Lightning)
#          Small Launcher | UFO Construction | UFO Navigation | UFO Propulsion
#          Ultimate Craft (Avenger)
# 40 pts : Blaster Launcher | Elerium-115
# 50 pts : Alien Autopsies (each type) | Alien Interrogations (per new entry)
# 60 pts : Alien Origins | Cydonia or Bust | The Martian Solution
RESEARCH_SCORE_SCALE = 1.0  # 1 research-point = 1 reward unit

MAX_STEPS = 50_000   # per-episode truncation (1-min steps, 31 days ≈ 44 640)

END_NONE = 0
END_WIN  = 1
END_LOSE = 2

_CRAFT_STATUS_INT: dict[str, int] = {
    "STR_READY":      0,
    "STR_OUT":        1,
    "STR_REPAIRS":    2,
    "STR_REFUELLING": 2,
    "STR_REARMING":   2,
}

# ── bridge client ──────────────────────────────────────────────────────────────

class _BridgeClient:
    def __init__(self, socket_path: str) -> None:
        self._sock = _ws.open_socket(socket_path)
        self._buf  = b""

    def close(self) -> None:
        try:
            _ws.close_socket(self._sock)
        except Exception:
            pass

    def send(self, **kw) -> None:
        _ws.send_bytes(self._sock, (json.dumps(kw) + "\n").encode())

    def drain_buf(self) -> None:
        """Discard the in-memory receive buffer.

        The visible-mode geoscape timer broadcasts state every 10ms.  Over a
        long episode (thousands of steps) thousands of tick messages accumulate
        in _buf because each step only consumes one.  Call this before
        sending a command that expects a specific response (e.g. new_game_ready)
        so the search loop is not swamped by stale tick events.
        """
        self._buf = b""

    def recv_one(self, timeout: float = 10.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Drain any complete messages already in the local buffer
            # before waiting on the socket for more data.
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                text = line.strip()
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
            # Buffer exhausted — wait for the engine to send more.
            ms = max(1, int((deadline - time.time()) * 1000))
            if not _ws.recv_ready(self._sock, timeout_ms=min(ms, 500)):
                continue
            chunk = _ws.recv_bytes(self._sock)
            if chunk is None:
                raise ConnectionError("engine closed bridge socket")
            self._buf += chunk
        return None

    def next_state(self, timeout: float = 15.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv_one(timeout=max(0.05, deadline - time.time()))
            if msg is None:
                break
            if msg.get("type") == "state":
                return msg
        return None


# ── environment ────────────────────────────────────────────────────────────────

class XcomInterceptionEnv(gym.Env):
    """
    OpenXcom strategic-layer episode driven entirely by simulated GUI clicks.

    Observation space (Dict):
      funds            Box int32 [0, 100 000 000]
      score            Box int32 [-10 000, 10 000]
      total_score      Box int32 [-100 000, 100 000]
      month            Box int32 [1, 12]
      day              Box int32 [1, 31]
      hour             Box int32 [0, 23]
      craft_status     Discrete(3)    0=ready 1=out 2=repair/refuel
      craft_fuel_pct   Box int32 [0, 100]
      craft_damage_pct Box int32 [0, 100]
      ufo_detected     Discrete(2)    1 when ≥1 UFO on radar
      mission_site     Discrete(2)    1 when a terror/mission site exists
      popup_depth      Box int32 [0, 10]
      screen_type      Discrete(11)   which dialog is on top (see SCREEN_TYPES)
      ending           Discrete(3)    0=ongoing 1=won 2=lost
      num_soldiers     Box int32 [0, 50]

    Action space: Discrete(23)  — see ACTIONS table above
    """

    metadata = {"render_modes": []}

    # Reward constants
    SCORE_SCALE          = 0.001  # routine strategic score ticks (small deltas)
    BATTLE_SCORE_SCALE   = 1.0   # post-battle total_score delta (§4.4)
    RESEARCH_SCORE_SCALE = 1.0   # research completion score delta (§4.8)
    CLICK_REWARD         = 0.002  # any mouse click (named button or random_click)
    UFO_BONUS            = 10.0   # first tick with a detected UFO
    LAUNCH_BONUS         = 50.0   # craft transitions to STR_OUT
    CYDONIA_BONUS        = 1000.0
    MENU_ESCAPE_REWARD   = 5.0    # screen transitions from main_menu → geoscape

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        *,
        headless: bool = True,
        socket_name: str = "openxcom_bridge.sock",
        speed_ms: int = 0,
        prefill_ufo: bool = False,   # no-op — kept for API compatibility
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Dict({
            "funds":            spaces.Box(0, 100_000_000, shape=(1,), dtype=np.int32),
            "score":            spaces.Box(-10_000, 10_000, shape=(1,), dtype=np.int32),
            "total_score":      spaces.Box(-100_000, 100_000, shape=(1,), dtype=np.int32),
            "month":            spaces.Box(1, 12,  shape=(1,), dtype=np.int32),
            "day":              spaces.Box(1, 31,  shape=(1,), dtype=np.int32),
            "hour":             spaces.Box(0, 23,  shape=(1,), dtype=np.int32),
            "craft_status":     spaces.Discrete(3),
            "craft_fuel_pct":   spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "craft_damage_pct": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "ufo_detected":     spaces.Discrete(2),
            "mission_site":     spaces.Discrete(2),
            "popup_depth":      spaces.Box(0, 10,  shape=(1,), dtype=np.int32),
            "screen_type":      spaces.Discrete(N_SCREEN_TYPES),
            "ending":           spaces.Discrete(3),
            "num_soldiers":     spaces.Box(0, 50,  shape=(1,), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode  = render_mode
        self.max_steps    = max_steps

        self._headless    = headless
        self._socket_name = socket_name
        self._speed_ms    = speed_ms

        self._proc:             Optional[subprocess.Popen] = None
        self._client:           Optional[_BridgeClient]    = None
        self._last_state:       Optional[Dict]             = None
        self._total_score_prev:    int   = 0
        self._research_score_prev: int   = 0
        self._step_count:          int   = 0
        self._ep_reward:           float = 0.0
        self._prev_craft_out:      bool  = False
        self._prev_ufo_seen:       bool  = False
        self._prev_at_menu:        bool  = False

        try:
            rank = int(socket_name.replace("openxcom_bridge_", "").replace(".sock", ""))
        except ValueError:
            rank = 0
        self._rank        = rank
        self._episode_num = 0
        os.makedirs(_LOG_DIR, exist_ok=True)
        self._log_path = os.path.join(_LOG_DIR, f"env_{rank}.jsonl")

        atexit.register(self._teardown)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if self._proc is not None and self._client is not None:
            if not self._new_game():
                self._teardown()
                self._launch_engine()
        else:
            self._teardown()
            self._launch_engine()

        state = self._last_state
        game  = state.get("game", {})
        self._total_score_prev    = game.get("total_score", 0)
        self._research_score_prev = game.get("score", 0)
        self._step_count          = 0
        self._ep_reward           = 0.0
        if self._client is not None:
            self._client.send(action="set_reward", value=0)
        self._prev_craft_out      = False
        self._prev_ufo_seen       = False
        self._prev_at_menu        = (game.get("screen_hint") == "main_menu")
        self._episode_num        += 1
        return self._build_obs(state), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._client is None or self._last_state is None:
            raise RuntimeError("call reset() before step()")

        total_score_before = self._total_score_prev

        try:
            state = self._execute_action(action)
        except ConnectionError:
            self._teardown()
            obs = self._build_obs(self._last_state) if self._last_state else self.observation_space.sample()
            self._ep_reward -= 10.0
            self._log_episode(self._last_state or {}, END_NONE, True, False)
            return obs, -10.0, True, False, {"error": "engine_disconnected"}

        self._last_state   = state
        self._step_count  += 1
        game               = state.get("game", {})
        total_score_now    = game.get("total_score", 0)
        research_score_now = game.get("score", 0)
        ending             = game.get("ending", END_NONE)

        # ── reward components ──────────────────────────────────────────────────
        # Mouse clicks (named buttons + random_click) earn CLICK_REWARD.
        # wait and dismiss are keyboard/no-op actions and earn nothing here.
        name, x, _y, _b = ACTIONS[action]
        is_click = (x is not None and x >= 0) or (name in ("random_click", "random_right_click"))
        reward = self.CLICK_REWARD if is_click else 0.0

        # Strategic score: large deltas are post-battle (§4.4), small are routine
        score_delta = total_score_now - total_score_before
        if abs(score_delta) >= BATTLE_SCORE_THRESHOLD:
            reward += score_delta * self.BATTLE_SCORE_SCALE
        else:
            reward += score_delta * self.SCORE_SCALE

        # Research completion (§4.8): score only ever increases; any positive
        # delta means a project just finished. Scale at 1 point = 1 reward unit.
        research_delta            = research_score_now - self._research_score_prev
        self._total_score_prev    = total_score_now
        self._research_score_prev = research_score_now
        if research_delta > 0:
            reward += research_delta * self.RESEARCH_SCORE_SCALE

        # UFO first-detection bonus
        ufo_now = bool(state.get("ufos") and any(u.get("detected") for u in state["ufos"]))
        if ufo_now and not self._prev_ufo_seen:
            reward += self.UFO_BONUS
        self._prev_ufo_seen = ufo_now

        # Craft-launched bonus: any craft transitions to STR_OUT
        craft_out_now = self._any_craft_out(state)
        if craft_out_now and not self._prev_craft_out:
            reward += self.LAUNCH_BONUS
        self._prev_craft_out = craft_out_now

        # Menu-escape bonus: main_menu → geoscape transition
        at_menu_now = (game.get("screen_hint") == "main_menu")
        if self._prev_at_menu and not at_menu_now:
            reward += self.MENU_ESCAPE_REWARD
        self._prev_at_menu = at_menu_now

        terminated = False
        if ending == END_WIN:
            reward    += self.CYDONIA_BONUS
            terminated = True
        elif ending == END_LOSE:
            terminated = True

        truncated = bool(self._step_count >= self.max_steps)

        self._ep_reward += reward
        self._client.send(action="set_reward", value=int(self._ep_reward))
        if terminated or truncated:
            self._log_episode(state, ending, terminated, truncated)

        return self._build_obs(state), reward, terminated, truncated, {}

    def close(self) -> None:
        self._teardown()

    # ── action execution ───────────────────────────────────────────────────────

    def _execute_action(self, action: int) -> Dict:
        name, x, y, button = ACTIONS[action]

        if name == "wait":
            if self._headless:
                self._client.send(action="step")
                return self._client.next_state(timeout=10.0) or self._last_state
            else:
                # Visible: poll for next broadcast from the running timer
                return self._client.next_state(timeout=0.5) or self._last_state

        if name == "menu_new_game":
            # Only acts when screen_type is main_menu; otherwise behaves like wait.
            at_menu = (self._last_state or {}).get("game", {}).get("screen_hint") == "main_menu"
            if at_menu:
                self._client.send(action="new_game")
                # cmdNewGame() broadcasts "new_game_ready" immediately.
                # In headless mode also send "step" as a fallback trigger.
                if self._headless:
                    self._client.send(action="step")
                return self._client.next_state(timeout=10.0) or self._last_state
            else:
                # Not at menu: no-op, just wait for next geoscape tick
                if self._headless:
                    self._client.send(action="step")
                    return self._client.next_state(timeout=10.0) or self._last_state
                else:
                    return self._client.next_state(timeout=0.5) or self._last_state
        elif name == "dismiss":
            self._client.send(action="dismiss")
        elif name == "key_a":
            self._client.send(action="key", key="a")
        elif name == "key_enter":
            self._client.send(action="key", key="enter")
        else:
            # Mouse click (named button or random position)
            if x == _RANDOM_CLICK_SENTINEL:
                x = int(self.np_random.integers(0, SCREEN_W))
                y = int(self.np_random.integers(0, SCREEN_H))
            self._client.send(action="click", x=x, y=y, button=button)

        if self._headless:
            # In headless mode only "step" triggers a state broadcast.
            # Sending one here makes the engine advance time and emit a fresh
            # "tick" after processing the action above, avoiding the 2 s timeout
            # that would otherwise fire for every click/dismiss/key action.
            self._client.send(action="step")
            return self._client.next_state(timeout=10.0) or self._last_state
        else:
            timeout = 0.5 if name in ("dismiss", "key_a", "key_enter") else 0.3
            return self._client.next_state(timeout=timeout) or self._last_state

    # ── engine lifecycle ───────────────────────────────────────────────────────

    def _launch_engine(self) -> None:
        socket_path = os.path.join(_TEMP, self._socket_name)

        env = os.environ.copy()
        # Headless engines use XCOM_BRIDGE_AUTOSTART to skip the main menu and
        # jump directly to geoscape.  Visible engines intentionally show the main
        # menu so the agent can learn to navigate it (Option 3).
        if self._headless:
            env["XCOM_BRIDGE_AUTOSTART"] = "1"
        env["XCOM_BRIDGE_SOCKET"]    = self._socket_name
        if self._speed_ms > 0:
            env["XCOM_GAME_SPEED"] = str(self._speed_ms)

        args = [ENGINE_EXE]
        if self._headless:
            args.append("--headless")

        self._proc = subprocess.Popen(
            args, cwd=ENGINE_CWD, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        for _ in range(100):
            time.sleep(0.1)
            try:
                self._client = _BridgeClient(socket_path)
                break
            except (OSError, ConnectionRefusedError):
                pass
        else:
            raise RuntimeError(f"bridge socket {socket_path!r} did not open within 10s")

        first = None
        if self._headless:
            for _ in range(5):
                self._client.send(action="step")
                first = self._client.next_state(timeout=15.0)
                if first is not None:
                    break
        else:
            # Visible mode: engine loads (~15s) then shows the main menu.
            # autoStart() broadcasts a "main_menu" state every ~100ms once loading
            # is done.  Just wait — the agent will navigate from here via
            # menu_new_game.  Allow up to 120s for slow machines.
            first = self._client.next_state(timeout=120.0)

        if first is None:
            raise RuntimeError("no initial state received from engine")

        self._last_state       = first
        self._total_score_prev = first.get("game", {}).get("total_score", 0)
        self._prev_craft_out   = False
        self._prev_ufo_seen    = False

    def _new_game(self) -> bool:
        try:
            # Discard stale tick events accumulated from the previous episode.
            # Without this, thousands of buffered ticks swamp the 5-iteration
            # loop and new_game_ready is never seen → _teardown() fires → window
            # closes and the 15s engine reload freeze begins.
            self._client.drain_buf()
            self._client.send(action="new_game")
            # Time-based loop: new_game_ready arrives within one geoscape frame
            # (~10ms) in visible mode or immediately via cmdNewGame() in headless.
            deadline = time.time() + 15.0
            while time.time() < deadline:
                state = self._client.next_state(timeout=min(5.0, deadline - time.time()))
                if state is None:
                    break
                if state.get("event") == "new_game_ready":
                    self._last_state       = state
                    self._total_score_prev = state.get("game", {}).get("total_score", 0)
                    self._prev_craft_out   = False
                    self._prev_ufo_seen    = False
                    return True
        except Exception:
            pass
        return False

    def _teardown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None
            time.sleep(0.3)

    def _log_episode(self, state: Dict, ending: int, terminated: bool, truncated: bool) -> None:
        game  = state.get("game", {})
        entry = {
            "t":           time.time(),
            "env":         self._rank,
            "ep":          self._episode_num,
            "steps":       self._step_count,
            "reward":      round(self._ep_reward, 3),
            "total_score": game.get("total_score", 0),
            "ending":      ending,
            "truncated":   truncated,
            "month":       game.get("month", 1),
            "day":         game.get("day",   1),
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── observation builder ────────────────────────────────────────────────────

    def _build_obs(self, state: Dict) -> Dict[str, Any]:
        g      = state.get("game", {})
        bases  = state.get("bases", [])
        craft  = self._first_craft(state) or {}
        fm     = max(1, craft.get("fuel_max",   1))
        dm     = max(1, craft.get("damage_max", 1))
        cs     = _CRAFT_STATUS_INT.get(craft.get("status", "STR_READY"), 0)

        ufos        = state.get("ufos", [])
        ufo_det     = int(any(u.get("detected") for u in ufos))
        mission_det = int(bool(state.get("mission_sites")))
        num_sol     = sum(len(b.get("soldiers", [])) for b in bases)

        screen_hint = g.get("screen_hint", "geoscape")
        screen_int  = SCREEN_TYPES.get(screen_hint, SCREEN_TYPES["other"])

        return {
            "funds":            np.array([int(np.clip(g.get("funds", 0), 0, 100_000_000))], dtype=np.int32),
            "score":            np.array([int(np.clip(g.get("score", 0), -10_000, 10_000))], dtype=np.int32),
            "total_score":      np.array([int(np.clip(g.get("total_score", 0), -100_000, 100_000))], dtype=np.int32),
            "month":            np.array([int(np.clip(g.get("month", 1), 1, 12))], dtype=np.int32),
            "day":              np.array([int(np.clip(g.get("day",   1), 1, 31))], dtype=np.int32),
            "hour":             np.array([int(np.clip(g.get("hour",  0), 0, 23))], dtype=np.int32),
            "craft_status":     np.int64(cs),
            "craft_fuel_pct":   np.array([int(np.clip(100 * craft.get("fuel", 0) // fm, 0, 100))], dtype=np.int32),
            "craft_damage_pct": np.array([int(np.clip(100 * craft.get("damage", 0) // dm, 0, 100))], dtype=np.int32),
            "ufo_detected":     np.int64(ufo_det),
            "mission_site":     np.int64(mission_det),
            "popup_depth":      np.array([int(np.clip(g.get("popup_depth", 0), 0, 10))], dtype=np.int32),
            "screen_type":      np.int64(screen_int),
            "ending":           np.int64(int(g.get("ending", END_NONE))),
            "num_soldiers":     np.array([int(np.clip(num_sol, 0, 50))], dtype=np.int32),
        }

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _first_craft(state: Dict) -> Optional[Dict]:
        bases = state.get("bases", [])
        if not bases:
            return None
        crafts = bases[0].get("crafts", [])
        return crafts[0] if crafts else None

    @staticmethod
    def _any_craft_out(state: Dict) -> bool:
        bases = state.get("bases", [])
        for b in bases:
            for c in b.get("crafts", []):
                if c.get("status") == "STR_OUT":
                    return True
        return False
