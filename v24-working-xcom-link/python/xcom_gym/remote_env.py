"""
RemoteXcomEnv — gym.Env proxy for an env served by xcom_rl.env_server.

Runs on the learner machine (A).  Each instance opens one TCP connection to an
env_server slot on the farm machine (B); step()/reset() are blocking JSON
round-trips.  Spaces are rebuilt locally from xcom_gym.env (both machines must
run the same commit — the hello handshake checks N_ACTIONS to catch skew).

Used via train_ppo's --remote HOST:PORT:COUNT flag; proxies drop into
SubprocVecEnv exactly like local envs.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from xcom_gym.env import N_ACTIONS

_CONNECT_RETRIES = 5
_CONNECT_BACKOFF = 3.0    # seconds between connect attempts
_IO_TIMEOUT      = 120.0  # per-request timeout; quick-battle resets can be slow


class RemoteXcomEnv(gym.Env):
    """TCP proxy to a remote XcomInterceptionEnv (see xcom_rl.env_server)."""

    metadata = {"render_modes": []}

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._buf  = b""

        # Same spaces as the served env; built locally without engine launch.
        self.observation_space = self._build_spaces_from_template()
        self.action_space      = gym.spaces.Discrete(N_ACTIONS)

    @staticmethod
    def _build_spaces_from_template() -> gym.spaces.Dict:
        # Instantiating the real env just for its spaces would launch nothing
        # (engine starts on reset), but carries ctor side effects (log files);
        # replicate the space dict from the class definition instead.
        from gymnasium import spaces
        from xcom_gym.env import N_SCREEN_TYPES
        return spaces.Dict({
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
            "alien_bases_found": spaces.Box(0, 20, shape=(1,), dtype=np.int32),
            "base_grid":        spaces.Box(0, 2,   shape=(36,), dtype=np.int8),
        })

    # ── wire protocol ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(_CONNECT_RETRIES):
            try:
                self._sock = socket.create_connection((self._host, self._port), timeout=10.0)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock.settimeout(_IO_TIMEOUT)
                self._buf = b""
                hello = self._request({"cmd": "hello"})
                if hello.get("n_actions") != N_ACTIONS:
                    raise RuntimeError(
                        f"env_server at {self._host}:{self._port} has N_ACTIONS="
                        f"{hello.get('n_actions')}, local N_ACTIONS={N_ACTIONS} — "
                        f"machines are on different commits"
                    )
                return
            except (OSError, ConnectionError) as exc:
                last_exc = exc
                self._close_sock()
                time.sleep(_CONNECT_BACKOFF)
        raise ConnectionError(
            f"could not connect to env_server {self._host}:{self._port} "
            f"after {_CONNECT_RETRIES} attempts: {last_exc}"
        )

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            raise ConnectionError("not connected")
        self._sock.sendall((json.dumps(payload) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("env_server closed connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"env_server error: {resp['error']}")
        return resp

    def _deserialize_obs(self, obs: dict) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, space in self.observation_space.spaces.items():
            v = obs[k]
            if isinstance(space, gym.spaces.Discrete):
                out[k] = np.int64(v)
            else:
                out[k] = np.asarray(v, dtype=space.dtype)
        return out

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if self._sock is None:
            self._connect()
        try:
            resp = self._request({"cmd": "reset"})
        except (ConnectionError, OSError, socket.timeout):
            # One reconnect attempt — the server may have recycled the session.
            self._close_sock()
            self._connect()
            resp = self._request({"cmd": "reset"})
        return self._deserialize_obs(resp["obs"]), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        resp = self._request({"cmd": "step", "action": int(action)})
        return (
            self._deserialize_obs(resp["obs"]),
            float(resp["reward"]),
            bool(resp["terminated"]),
            bool(resp["truncated"]),
            {},
        )

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._request({"cmd": "close"})
            except (ConnectionError, OSError, RuntimeError, socket.timeout):
                pass
            self._close_sock()
