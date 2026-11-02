"""Background thread that connects to the OpenXcom bridge socket and
fans out every incoming JSON line to all active WebSocket subscribers.

Threading model
---------------
The xcom_bridge._winsock module is built on blocking ctypes Winsock2 calls
(no asyncio support), so the socket reader must live in a plain OS thread.
Fan-out to asyncio consumers is done through per-subscriber asyncio.Queues;
loop.call_soon_threadsafe delivers messages without holding the GIL longer
than necessary.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Optional, Set

# Allow running from this directory or from the python/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xcom_bridge import _winsock as _ws
from .aggregator import compute_metrics

SOCK_PATH: str = os.path.join(
    os.environ.get("TEMP", r"C:\Windows\Temp"),
    "openxcom_bridge.sock",
)
RECONNECT_DELAY: float = 3.0   # seconds before retry on disconnect
RECV_TIMEOUT_MS: int   = 200   # select() poll interval inside the read loop
QUEUE_MAXSIZE:   int   = 512   # per-client queue depth; old messages dropped if full


# ---------------------------------------------------------------------------
# Broadcast manager
# ---------------------------------------------------------------------------

class BroadcastManager:
    """Thread-safe fan-out from one reader thread to N async WebSocket queues."""

    def __init__(self) -> None:
        self._lock:   threading.Lock                         = threading.Lock()
        self._queues: Set[asyncio.Queue[str]]                = set()
        self._loop:   Optional[asyncio.AbstractEventLoop]    = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the asyncio event loop.  Call once at app startup."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[str]:
        """Create and return a new per-client queue; registers it for delivery."""
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        with self._lock:
            self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        """Deregister a queue when its WebSocket client disconnects."""
        with self._lock:
            self._queues.discard(q)

    def broadcast(self, message: str) -> None:
        """Deliver *message* to every subscriber.  Called from the reader thread."""
        if self._loop is None or not self._loop.is_running():
            return
        with self._lock:
            targets = list(self._queues)
        for q in targets:
            # _deliver runs inside the event loop thread, so Queue ops are safe.
            self._loop.call_soon_threadsafe(self._deliver, q, message)

    @staticmethod
    def _deliver(q: asyncio.Queue[str], message: str) -> None:
        """Drop-oldest-then-insert when the queue is at capacity."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            pass  # shouldn't happen after the drain above, but guard anyway


# Module-level singleton shared by app.py and the reader thread.
broadcast_manager = BroadcastManager()


# ---------------------------------------------------------------------------
# Reader thread
# ---------------------------------------------------------------------------

def _reader_loop(stop: threading.Event) -> None:
    """Outer reconnect loop — runs until *stop* is set."""
    while not stop.is_set():
        try:
            print(f"[bridge-reader] Connecting to {SOCK_PATH} …", flush=True)
            sock = _ws.open_socket(SOCK_PATH)
        except (ConnectionRefusedError, OSError) as exc:
            print(f"[bridge-reader] {exc}  (retry in {RECONNECT_DELAY:.0f}s)", flush=True)
            stop.wait(timeout=RECONNECT_DELAY)
            continue

        print("[bridge-reader] Connected.", flush=True)
        buf = b""
        try:
            while not stop.is_set():
                if not _ws.recv_ready(sock, timeout_ms=RECV_TIMEOUT_MS):
                    continue
                chunk = _ws.recv_bytes(sock, 65536)
                if chunk is None:
                    print("[bridge-reader] Engine closed the connection.", flush=True)
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.strip().decode("utf-8", errors="replace")
                    if not text:
                        continue
                    # Silently drop malformed JSON rather than crashing.
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    broadcast_manager.broadcast(text)
                    # Immediately follow with aggregated metrics for geoscape state msgs.
                    if obj.get("type") == "state":
                        try:
                            agg = json.dumps(compute_metrics(obj), separators=(",", ":"))
                            broadcast_manager.broadcast(agg)
                        except Exception:
                            pass  # never let aggregation crash the reader loop
        finally:
            _ws.close_socket(sock)
            print("[bridge-reader] Disconnected.", flush=True)

        if not stop.is_set():
            stop.wait(timeout=RECONNECT_DELAY)


_stop_event: threading.Event          = threading.Event()
_thread:     Optional[threading.Thread] = None


def start() -> None:
    """Spawn the reader thread.  Call once from the FastAPI lifespan."""
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(
        target=_reader_loop,
        args=(_stop_event,),
        daemon=True,
        name="bridge-reader",
    )
    _thread.start()


def stop() -> None:
    """Signal the reader thread to exit and wait for it to finish."""
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5.0)
