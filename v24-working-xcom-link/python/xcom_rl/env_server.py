"""
env_server.py — TCP env farm for distributed rollout collection.

Runs on a secondary machine (B) and serves XcomInterceptionEnv instances over
plain TCP to the PPO learner on the primary machine (A).  One env (and one
engine process) per client connection.  The learner connects via
xcom_gym.remote_env.RemoteXcomEnv, added to training with train_ppo's
--remote HOST:PORT:COUNT flag.

Protocol: newline-delimited JSON, one request → one response.
  {"cmd": "hello"}                  → {"n_actions": 7429, "version": 1}
  {"cmd": "reset"}                  → {"obs": {...}}
  {"cmd": "step", "action": 42}     → {"obs": {...}, "reward": 0.01,
                                       "terminated": false, "truncated": false}
  {"cmd": "close"}                  → {"ok": true}   (connection then closes)

Observations are serialized as plain lists / ints; the client reconstructs
dtypes from its local copy of the observation space.  Both machines must run
the same commit — the hello handshake checks N_ACTIONS to catch skew.

Usage (machine B, from python/ directory):
  py -3 -m xcom_rl.env_server --port 7100 --n-envs 3 --battle-only

Notes:
- Battle-only envs run in visible mode (GUI click-through of the New Battle
  menu); expect one game window per env on this machine.
- Allow the port through Windows Firewall:
    netsh advfirewall firewall add rule name="xcom_env_server" dir=in
          action=allow protocol=TCP localport=7100
- This machine runs no learner and must not write the checkpoint registry.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import xcom_gym  # noqa: F401
from xcom_gym.env import XcomInterceptionEnv, N_ACTIONS

PROTOCOL_VERSION = 1

# Bridge sockets for served envs use ranks 200+ so they never collide with
# local training (0-99) or eval (100+) envs if this machine also runs those.
_SERVER_RANK_OFFSET = 200


def _serialize_obs(obs: dict) -> dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = int(v)   # Discrete sub-spaces arrive as numpy int64 scalars
    return out


class _ClientSession(threading.Thread):
    """One connected learner proxy = one env = one engine process."""

    def __init__(self, conn: socket.socket, addr, rank: int, env_kwargs: dict) -> None:
        super().__init__(daemon=True)
        self._conn = conn
        self._addr = addr
        self._rank = rank
        self._env_kwargs = env_kwargs
        self._env: XcomInterceptionEnv | None = None

    def _send(self, payload: dict) -> None:
        self._conn.sendall((json.dumps(payload) + "\n").encode())

    def _handle(self, req: dict) -> bool:
        """Process one request; returns False when the session should end."""
        cmd = req.get("cmd")
        if cmd == "hello":
            self._send({"n_actions": N_ACTIONS, "version": PROTOCOL_VERSION})
        elif cmd == "reset":
            if self._env is None:
                self._env = XcomInterceptionEnv(
                    socket_name=f"openxcom_bridge_{_SERVER_RANK_OFFSET + self._rank}.sock",
                    **self._env_kwargs,
                )
            obs, _info = self._env.reset()
            self._send({"obs": _serialize_obs(obs)})
        elif cmd == "step":
            if self._env is None:
                self._send({"error": "step before reset"})
                return False
            obs, reward, terminated, truncated, _info = self._env.step(int(req["action"]))
            self._send({
                "obs":        _serialize_obs(obs),
                "reward":     float(reward),
                "terminated": bool(terminated),
                "truncated":  bool(truncated),
            })
        elif cmd == "close":
            self._send({"ok": True})
            return False
        else:
            self._send({"error": f"unknown cmd {cmd!r}"})
        return True

    def run(self) -> None:
        print(f"[env_server] rank {self._rank}: client {self._addr} connected")
        buf = b""
        try:
            while True:
                while b"\n" not in buf:
                    chunk = self._conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                if not self._handle(json.loads(line)):
                    return
        except (ConnectionError, OSError):
            pass   # learner disconnected; clean up below
        except Exception:
            print(f"[env_server] rank {self._rank}: crashed:\n{traceback.format_exc()}")
            try:
                self._send({"error": "env crashed; see server log"})
            except OSError:
                pass
        finally:
            print(f"[env_server] rank {self._rank}: client {self._addr} disconnected")
            if self._env is not None:
                self._env.close()
                self._env = None
            try:
                self._conn.close()
            except OSError:
                pass


def serve(port: int, n_envs: int, env_kwargs: dict) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(n_envs)
    mode = "battle-only" if env_kwargs.get("battle_only") else "campaign"
    print(f"[env_server] listening on 0.0.0.0:{port}  ({n_envs} env slots, {mode} mode)")
    print(f"[env_server] N_ACTIONS={N_ACTIONS}  protocol v{PROTOCOL_VERSION}")

    rank = 0
    sessions: list[_ClientSession] = []
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sessions = [s for s in sessions if s.is_alive()]
            if len(sessions) >= n_envs:
                print(f"[env_server] rejecting {addr}: all {n_envs} slots busy")
                conn.close()
                continue
            session = _ClientSession(conn, addr, rank % n_envs, env_kwargs)
            sessions.append(session)
            session.start()
            rank += 1
    except KeyboardInterrupt:
        print("\n[env_server] shutting down")
    finally:
        srv.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve XcomInterceptionEnv instances over TCP")
    p.add_argument("--port",        type=int, default=7100)
    p.add_argument("--n-envs",      type=int, default=3,
                   help="Max concurrent envs (one engine process each)")
    p.add_argument("--battle-only", action="store_true",
                   help="Serve quick-battle episodes (New Battle mode, visible windows)")
    p.add_argument("--max-steps",   type=int, default=0,
                   help="Per-episode step cap (0 = mode default)")
    p.add_argument("--speed-ms",    type=int, default=10,
                   help="Engine tick interval ms for visible envs (keep low: rollouts are lockstep)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kwargs: dict = {"battle_only": args.battle_only, "speed_ms": args.speed_ms}
    if not args.battle_only:
        kwargs["headless"] = True
    if args.max_steps > 0:
        kwargs["max_steps"] = args.max_steps
    serve(args.port, args.n_envs, kwargs)
