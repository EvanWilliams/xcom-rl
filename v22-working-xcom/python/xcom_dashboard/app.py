"""FastAPI application: XCom Bridge Dashboard.

Endpoints
---------
GET  /        — Dashboard SPA (served from static/index.html)
GET  /static/ — Static file tree
WS   /ws      — Bridge relay; frames are raw JSON + aggregated metrics
GET  /status  — JSON health check

Usage
-----
    # From python/
    py -m uvicorn xcom_dashboard.app:app
    py -m xcom_dashboard          # shortcut
"""

from __future__ import annotations

import asyncio
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bridge_reader
from .bridge_reader import broadcast_manager, SOCK_PATH

STATIC_DIR   = pathlib.Path(__file__).parent / "static"
KEEPALIVE_SECS = 25


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcast_manager.set_loop(asyncio.get_running_loop())
    bridge_reader.start()
    yield
    bridge_reader.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="XCom Dashboard", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# HTTP routes  (must be registered before the static catch-all mount)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the dashboard SPA."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/status")
async def status() -> JSONResponse:
    thread = bridge_reader._thread
    return JSONResponse({
        "bridge_socket": SOCK_PATH,
        "reader_alive":  thread is not None and thread.is_alive(),
        "subscribers":   len(broadcast_manager._queues),
    })


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_relay(ws: WebSocket) -> None:
    """Relay all bridge messages (raw state + aggregated metrics) to the client."""
    await ws.accept()
    q = broadcast_manager.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECS)
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"keepalive"}')
                continue
            await ws.send_text(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        broadcast_manager.unsubscribe(q)


# ---------------------------------------------------------------------------
# Static files  (catch-all, must come last)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
