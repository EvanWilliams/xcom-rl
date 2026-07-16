"""
Gymnasium Env wrapping a single OpenXcom episode.

reset()  – reuses or restarts the engine; optionally fast-forwards to first UFO.
step(0)  – wait: advance one 5-second tick (headless) or drain latest state (visible).
step(1)  – launch: send ready craft at detected UFO (falls back to wait if unavailable).
step(2)  – dismiss: send Escape to close any popup currently on screen.

Auto-dismiss: after each step, if popup_depth > 0, the env also sends dismiss so
agents never get permanently blocked by UI dialogs.

Terminal: craft launched (phase==2), or game ending detected (Cydonia win/loss).
Truncated: max_steps exceeded.

Reward:
  total_score_delta * SCORE_SCALE        (continuous strategic feedback)
  + PHASE_BONUS[phase_delta]             (+10 for ufo_visible, +50 for launch)
  + CYDONIA_BONUS if ending == END_WIN   (+1000, terminates episode)
  - STEP_PENALTY per step                (-0.01)
"""
from __future__ import annotations

import atexit
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

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

# ── tunable defaults ───────────────────────────────────────────────────────────

MAX_STEPS     = 3_000   # truncation limit per episode (headless)
MAX_SCAN_DAYS = 60      # in-game days to fast-forward scanning for a UFO
FF_CHUNK      = 5       # fast-forward step size in days

_CRAFT_STATUS_INT: Dict[str, int] = {
    "STR_READY":      0,
    "STR_OUT":        1,
    "STR_REPAIRS":    2,
    "STR_REFUELLING": 2,
    "STR_REARMING":   2,
}

END_NONE = 0
END_WIN  = 1
END_LOSE = 2


# ── minimal bridge client ──────────────────────────────────────────────────────

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

    def recv_one(self, timeout: float = 10.0) -> Optional[Dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ms = max(1, int((deadline - time.time()) * 1000))
            if not _ws.recv_ready(self._sock, timeout_ms=min(ms, 500)):
                continue
            chunk = _ws.recv_bytes(self._sock)
            if chunk is None:
                raise ConnectionError("engine closed the bridge socket")
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

    def next_state(self, timeout: float = 15.0) -> Optional[Dict]:
        """Drain messages until one with type=='state' arrives or timeout."""
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
    OpenXcom strategic layer episode.

    Observation space (Dict):
      funds            Box int32 [0, 100 000 000]    player funds
      score            Box int32 [-10 000, 10 000]   research score
      total_score      Box int32 [-100 000, 100 000] strategic score (xcom-alien delta)
      month            Box int32 [1, 12]
      day              Box int32 [1, 31]
      craft_status     Discrete(3)    0=ready 1=out 2=repair/refuel
      craft_fuel_pct   Box int32 [0, 100]
      craft_damage_pct Box int32 [0, 100]
      ufo_detected     Discrete(2)    1 when a UFO is on radar
      mission_phase    Discrete(4)    0=scan 1=ufo_visible 2=craft_out 3=returned
      popup_depth      Box int32 [0, 10]   UI dialog depth (0 = no popup)
      ending           Discrete(3)    0=ongoing 1=won 2=lost
      num_soldiers     Box int32 [0, 50]   soldiers available at base

    Action space: Discrete(3)
      0  wait / advance one 5-second tick (headless) or poll latest state (visible)
      1  launch the ready craft at the detected UFO (falls back to 0 if unavailable)
      2  dismiss: send Escape to close current popup dialog
    """

    metadata = {"render_modes": []}

    # Reward constants
    SCORE_SCALE    = 0.001    # total_score delta multiplier
    PHASE_BONUS    = {1: 10.0, 2: 50.0}   # bonus per phase reached (cumulative on transition)
    CYDONIA_BONUS  = 1000.0   # winning the game
    STEP_PENALTY   = 0.01     # cost per step to encourage efficiency

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        *,
        headless: bool = True,
        socket_name: str = "openxcom_bridge.sock",
        speed_ms: int = 0,
        prefill_ufo: bool = True,
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Dict({
            "funds":            spaces.Box(0, 100_000_000, shape=(1,), dtype=np.int32),
            "score":            spaces.Box(-10_000, 10_000, shape=(1,), dtype=np.int32),
            "total_score":      spaces.Box(-100_000, 100_000, shape=(1,), dtype=np.int32),
            "month":            spaces.Box(1, 12,  shape=(1,), dtype=np.int32),
            "day":              spaces.Box(1, 31,  shape=(1,), dtype=np.int32),
            "craft_status":     spaces.Discrete(3),
            "craft_fuel_pct":   spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "craft_damage_pct": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "ufo_detected":     spaces.Discrete(2),
            "mission_phase":    spaces.Discrete(4),
            "popup_depth":      spaces.Box(0, 10,  shape=(1,), dtype=np.int32),
            "ending":           spaces.Discrete(3),
            "num_soldiers":     spaces.Box(0, 50,  shape=(1,), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(3)
        self.render_mode  = render_mode
        self.max_steps    = max_steps

        self._headless    = headless
        self._socket_name = socket_name
        self._speed_ms    = speed_ms
        self._prefill_ufo = prefill_ufo

        self._proc:            Optional[subprocess.Popen] = None
        self._client:          Optional[_BridgeClient]    = None
        self._last_state:      Optional[Dict]             = None
        self._total_score_prev: int = 0
        self._step_count:      int  = 0
        self._phase:           int  = 0
        self._popup_steps:     int  = 0   # consecutive steps spent on a popup
        self._craft_name:      Optional[str] = None
        self._ufo_idx:         Optional[int] = None

        # Episode logging — extract rank from socket name for per-env log files
        try:
            rank = int(socket_name.replace("openxcom_bridge_", "").replace(".sock", ""))
        except ValueError:
            rank = 0
        self._rank        = rank
        self._episode_num = 0
        self._ep_reward   = 0.0
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

        if self._prefill_ufo and self._headless:
            state = self._scan_for_ufo()
        else:
            state = self._last_state

        self._last_state       = state
        self._total_score_prev = state.get("game", {}).get("total_score", 0)
        self._step_count       = 0
        self._ep_reward        = 0.0
        self._popup_steps      = 0
        self._episode_num     += 1
        return self._build_obs(state), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._client is None or self._last_state is None:
            raise RuntimeError("call reset() before step()")

        phase_before       = self._phase
        total_score_before = self._total_score_prev

        try:
            if not self._headless:
                state = self._step_visible(action)
            else:
                state = self._step_headless(action)

            # Dialog navigation state machine.
            # Handles three interceptable dialog types plus info-only dialogs:
            #
            #   UfoDetectedState  (flying/landed/crashed UFO):
            #     Intercept btn at (128,124) → InterceptState (depth+1)
            #   MissionDetectedState  (terror mission):
            #     Intercept btn at (128,138) → InterceptState (depth+1)
            #   InterceptState  (craft selection):
            #     First craft row at (152,82) → ConfirmDestinationState
            #   ConfirmDestinationState:
            #     OK btn at (93,110) → pops both, depth=0
            #
            #   LowFuelState / CraftErrorState / other info dialogs:
            #     No intercept buttons — Escape (dismiss) closes them via keyCancel.
            #
            # Sequence:
            #   steps  1-5:  alternate Intercept clicks for UFO (128,124) / Mission (128,138)
            #   steps  6-10: first craft row (152,82) for InterceptState
            #   steps 11-15: OK btn (93,110) for ConfirmDestinationState
            #   steps 16+:   dismiss (Escape) — closes info dialogs and cleans up
            #                failed intercepts (InterceptState/UfoDetected still open)
            #   steps 100+:  random button as last resort
            popup_depth = state.get("game", {}).get("popup_depth", 0)
            if popup_depth == 0:
                self._popup_steps = 0
            else:
                self._popup_steps += 1
                ps = self._popup_steps
                if ps > 100:
                    x, y = random.choice([(128, 124), (128, 138), (128, 140), (128, 156)])
                    self._client.send(action="click", x=x, y=y)
                elif ps > 15:
                    self._client.send(action="dismiss")
                else:
                    cycle = (ps - 1) % 15
                    if cycle < 5:
                        # Alternate: even=UfoDetectedState, odd=MissionDetectedState
                        if cycle % 2 == 0:
                            self._client.send(action="click", x=128, y=124)
                        else:
                            self._client.send(action="click", x=128, y=138)
                    elif cycle < 10:
                        self._client.send(action="click", x=152, y=82)
                    else:
                        self._client.send(action="click", x=93, y=110)
                nxt = self._client.next_state(timeout=0.5)
                if nxt is not None:
                    state = nxt

        except ConnectionError:
            # Engine process died (crash or OS kill). End the episode cleanly so
            # SB3 calls reset() which will relaunch the engine.
            self._teardown()
            obs = self._build_obs(self._last_state) if self._last_state else self.observation_space.sample()
            self._ep_reward -= 10.0   # penalty for dying
            self._log_episode(self._last_state or {}, END_NONE, True, False)
            return obs, -10.0, True, False, {"error": "engine_disconnected"}

        self._last_state       = state
        self._step_count      += 1
        game                   = state.get("game", {})
        total_score_now        = game.get("total_score", 0)
        self._total_score_prev = total_score_now
        ending                 = game.get("ending", END_NONE)

        # Phase-transition bonus
        phase_bonus = sum(
            self.PHASE_BONUS.get(p, 0.0)
            for p in range(phase_before + 1, self._phase + 1)
        )

        score_delta = total_score_now - total_score_before
        reward      = float(score_delta * self.SCORE_SCALE + phase_bonus - self.STEP_PENALTY)

        terminated = False
        if ending == END_WIN:
            reward    += self.CYDONIA_BONUS
            terminated = True
        elif ending == END_LOSE:
            terminated = True
        elif self._phase == 2:
            # Craft launched — end episode to avoid headless dogfight deadlock.
            # Visible mode: agent can keep playing past this point; change if needed.
            terminated = self._headless

        truncated = bool(self._step_count >= self.max_steps)

        self._ep_reward += reward
        if terminated or truncated:
            self._log_episode(state, ending, terminated, truncated)

        return self._build_obs(state), reward, terminated, truncated, {}

    def close(self) -> None:
        self._teardown()

    # ── step dispatch ──────────────────────────────────────────────────────────

    def _step_headless(self, action: int) -> Dict:
        if action == 2:
            # Only inject Escape when a popup is actually present. In headless mode
            # dismiss is a no-op at the engine level, but honour the guard anyway.
            popup_depth = (self._last_state or {}).get("game", {}).get("popup_depth", 0)
            if popup_depth > 0:
                self._client.send(action="dismiss")
                state = self._client.next_state(timeout=5.0) or self._last_state
                return state
            # No popup — treat as wait
            action = 0

        if (action == 1
                and self._phase == 1
                and self._craft_name is not None
                and self._ufo_idx is not None):
            self._client.send(
                action="launch_craft", base=0, craft=self._craft_name,
                target_type="ufo", target_id=self._ufo_idx,
            )
            state = self._client.next_state(timeout=5.0) or self._last_state
            self._phase = 2
        else:
            self._client.send(action="step")
            state = self._client.next_state(timeout=10.0) or self._last_state
            if self._phase < 2:
                self._refresh_targets(state)
        return state

    def _step_visible(self, action: int) -> Dict:
        if action == 2:
            # Only inject Escape when a popup is showing. Injecting Escape into the
            # bare Geoscape opens the pause menu — not what we want.
            popup_depth = (self._last_state or {}).get("game", {}).get("popup_depth", 0)
            if popup_depth > 0:
                self._client.send(action="dismiss")
                state = self._client.next_state(timeout=0.5) or self._last_state
                return state
            # No popup — fall through to normal poll

        if (action == 1
                and self._phase == 1
                and self._craft_name is not None
                and self._ufo_idx is not None):
            self._client.send(
                action="launch_craft", base=0, craft=self._craft_name,
                target_type="ufo", target_id=self._ufo_idx,
            )
            self._phase = 2
            state = self._client.next_state(timeout=0.5) or self._last_state
        else:
            state = self._client.next_state(timeout=0.15) or self._last_state
            self._refresh_targets(state)
        return state

    # ── engine lifecycle ───────────────────────────────────────────────────────

    def _launch_engine(self) -> None:
        socket_path = os.path.join(_TEMP, self._socket_name)

        env = os.environ.copy()
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
            raise RuntimeError(
                f"engine bridge socket {socket_path!r} did not open within 10 s"
            )

        if self._headless:
            first = None
            for _ in range(5):
                self._client.send(action="step")
                first = self._client.next_state(timeout=15.0)
                if first is not None:
                    break
        else:
            first = self._client.next_state(timeout=30.0)

        if first is None:
            raise RuntimeError("no initial state received from engine after launch")

        self._last_state       = first
        self._total_score_prev = first.get("game", {}).get("total_score", 0)
        self._phase            = 0
        self._popup_steps      = 0
        self._craft_name       = None
        self._ufo_idx          = None

    def _new_game(self) -> bool:
        assert self._client is not None and self._proc is not None
        try:
            self._client.send(action="new_game")
            for _ in range(5):
                state = self._client.next_state(timeout=15.0)
                if state is not None and state.get("event") == "new_game_ready":
                    self._last_state       = state
                    self._total_score_prev = state.get("game", {}).get("total_score", 0)
                    self._phase            = 0
                    self._popup_steps      = 0
                    self._craft_name       = None
                    self._ufo_idx          = None
                    return True
        except Exception:
            pass
        return False

    def _log_episode(self, state: Dict, ending: int, terminated: bool, truncated: bool) -> None:
        game = state.get("game", {})
        entry = {
            "t":           time.time(),
            "env":         self._rank,
            "ep":          self._episode_num,
            "steps":       self._step_count,
            "reward":      round(self._ep_reward, 3),
            "phase":       self._phase,
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

    # ── episode helpers ────────────────────────────────────────────────────────

    def _scan_for_ufo(self) -> Dict:
        """Fast-forward (headless) until one UFO is detected and one craft is ready."""
        assert self._client is not None
        state = self._last_state
        days  = 0

        while days < MAX_SCAN_DAYS:
            ufos   = [u for u in state.get("ufos", []) if u.get("detected")]
            crafts = [c for c in self._crafts(state) if c.get("status") == "STR_READY"]
            if ufos and crafts:
                self._ufo_idx    = ufos[0]["idx"]
                self._craft_name = crafts[0]["name"]
                self._phase      = 1
                return state

            self._client.send(action="fast_forward", days=FF_CHUNK)
            nxt = self._client.next_state(timeout=30.0)
            if nxt is None:
                break
            days += FF_CHUNK
            state = nxt

        self._phase      = 0
        self._ufo_idx    = None
        self._craft_name = None
        return state

    def _refresh_targets(self, state: Dict) -> None:
        ufos   = [u for u in state.get("ufos", []) if u.get("detected")]
        crafts = [c for c in self._crafts(state) if c.get("status") == "STR_READY"]
        if ufos and crafts:
            self._ufo_idx    = ufos[0]["idx"]
            self._craft_name = crafts[0]["name"]
            self._phase      = 1
        elif not ufos:
            self._ufo_idx = None
            self._phase   = 0

    # ── observation builder ────────────────────────────────────────────────────

    def _build_obs(self, state: Dict) -> Dict[str, Any]:
        g     = state.get("game", {})
        craft = self._find_craft(state) or {}
        bases = state.get("bases", [])

        fm  = max(1, craft.get("fuel_max",   1))
        dm  = max(1, craft.get("damage_max", 1))
        cs  = _CRAFT_STATUS_INT.get(craft.get("status", "STR_READY"), 0)

        num_soldiers = sum(len(b.get("soldiers", [])) for b in bases)

        return {
            "funds":            np.array([int(np.clip(g.get("funds",       0),    0, 100_000_000))], dtype=np.int32),
            "score":            np.array([int(np.clip(g.get("score",       0), -10_000, 10_000))],   dtype=np.int32),
            "total_score":      np.array([int(np.clip(g.get("total_score", 0), -100_000, 100_000))], dtype=np.int32),
            "month":            np.array([int(np.clip(g.get("month",       1),  1, 12))],            dtype=np.int32),
            "day":              np.array([int(np.clip(g.get("day",         1),  1, 31))],            dtype=np.int32),
            "craft_status":     np.int64(cs),
            "craft_fuel_pct":   np.array([int(np.clip(100 * craft.get("fuel",   0) // fm, 0, 100))], dtype=np.int32),
            "craft_damage_pct": np.array([int(np.clip(100 * craft.get("damage", 0) // dm, 0, 100))], dtype=np.int32),
            "ufo_detected":     np.int64(1 if self._ufo_idx is not None else 0),
            "mission_phase":    np.int64(self._phase),
            "popup_depth":      np.array([int(np.clip(g.get("popup_depth", 0), 0, 10))],            dtype=np.int32),
            "ending":           np.int64(int(g.get("ending", END_NONE))),
            "num_soldiers":     np.array([int(np.clip(num_soldiers, 0, 50))],                        dtype=np.int32),
        }

    # ── utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _crafts(state: Dict):
        bases = state.get("bases", [])
        return bases[0].get("crafts", []) if bases else []

    def _find_craft(self, state: Dict) -> Optional[Dict]:
        if self._craft_name is None:
            return None
        for c in self._crafts(state):
            if c.get("name") == self._craft_name:
                return c
        return None
