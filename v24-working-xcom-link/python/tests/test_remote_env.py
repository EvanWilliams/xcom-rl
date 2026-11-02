"""Tests for the TCP env farm layer (env_server + RemoteXcomEnv).

Runs without an engine: serialization round-trip tests use sampled
observations; the loopback test patches env_server's env class with a stub.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.remote_env import RemoteXcomEnv
from xcom_rl import env_server
from xcom_rl.env_server import _serialize_obs


def _spaces():
    return RemoteXcomEnv._build_spaces_from_template()


def test_serialize_roundtrip_preserves_dtypes_and_values():
    space = _spaces()
    proxy = RemoteXcomEnv.__new__(RemoteXcomEnv)
    proxy.observation_space = space

    obs = space.sample()
    restored = proxy._deserialize_obs(_serialize_obs(obs))

    assert set(restored) == set(obs)
    for k, sub in space.spaces.items():
        if hasattr(sub, "dtype") and hasattr(sub, "shape") and sub.shape:
            assert restored[k].dtype == sub.dtype, k
            assert restored[k].shape == sub.shape, k
            assert np.array_equal(restored[k], obs[k]), k
        else:  # Discrete → int64 scalar
            assert restored[k] == int(obs[k]), k
            assert isinstance(restored[k], np.int64), k
        assert sub.contains(restored[k]), k


class _StubEnv:
    """Engine-free stand-in for XcomInterceptionEnv inside env_server."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observation_space = _spaces()
        self.n_steps = 0

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        self.n_steps += 1
        term = self.n_steps % 5 == 0
        return self.observation_space.sample(), 0.25, term, False, {}

    def close(self):
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def loopback_server(monkeypatch):
    monkeypatch.setattr(env_server, "XcomInterceptionEnv", _StubEnv)
    port = _free_port()
    t = threading.Thread(
        target=env_server.serve, args=(port, 2, {"battle_only": True}), daemon=True,
    )
    t.start()
    time.sleep(0.3)  # let the listener bind
    return port


def test_loopback_reset_step_close(loopback_server):
    env = RemoteXcomEnv("127.0.0.1", loopback_server)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)

    saw_terminated = False
    for i in range(6):
        obs, reward, terminated, truncated, _ = env.step(i % env.action_space.n)
        assert env.observation_space.contains(obs)
        assert reward == pytest.approx(0.25)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        saw_terminated = saw_terminated or terminated
    assert saw_terminated  # stub terminates every 5th step
    env.close()


def test_handshake_rejects_wrong_n_actions(loopback_server, monkeypatch):
    import xcom_gym.remote_env as remote_mod
    monkeypatch.setattr(remote_mod, "N_ACTIONS", 123)
    monkeypatch.setattr(remote_mod, "_CONNECT_RETRIES", 1)
    monkeypatch.setattr(remote_mod, "_CONNECT_BACKOFF", 0.0)
    env = RemoteXcomEnv("127.0.0.1", loopback_server)
    with pytest.raises((ConnectionError, RuntimeError)):
        env.reset()
