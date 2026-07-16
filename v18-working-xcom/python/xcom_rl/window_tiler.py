"""
Tile OpenXcom training windows to maximise fully-visible count.

Pure-ctypes Win32 — no pywin32 dependency.
Safely no-ops on non-Windows platforms.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import time
from typing import List, Optional


def watch_and_retile(
    n: int,
    title_substr: str = "OpenXcom",
    check_interval: float = 2.0,
) -> None:
    """
    Long-running background thread entry point.

    1. Waits for *n* windows to appear and tiles them (initial layout).
    2. Keeps polling every *check_interval* seconds; whenever a new HWND
       appears (engine crashed and restarted), waits briefly for any other
       restarting windows then retiles the full set.

    Runs until the process exits (designed for a daemon thread).
    """
    if sys.platform != "win32":
        return

    # Initial tile.
    _do_tile(n, title_substr, timeout=30.0)
    known: set[int] = set(_find_windows_by_title(title_substr))

    # Watch for restarts.
    while True:
        time.sleep(check_interval)
        current: set[int] = set(_find_windows_by_title(title_substr))
        if current - known:
            # New HWND(s) detected — pause briefly for any other simultaneous
            # restarts, then retile everything.
            time.sleep(1.5)
            _do_tile(n, title_substr, timeout=15.0)
            known = set(_find_windows_by_title(title_substr))
        else:
            known = current


def tile_visible_windows(
    n: int,
    title_substr: str = "OpenXcom",
    timeout: float = 30.0,
) -> None:
    """One-shot tile: wait for *n* windows then arrange them into a grid."""
    if sys.platform != "win32":
        return
    _do_tile(n, title_substr, timeout)


def _do_tile(n: int, title_substr: str, timeout: float) -> None:
    """
    Core tile logic: wait for *n* windows, compute grid from work area,
    position each fully-visible slot.  Windows beyond the grid are left alone.
    """
    hwnds = _wait_for_windows(n, title_substr, timeout)
    if not hwnds:
        print("[window_tiler] No OpenXcom windows found within timeout; skipping tile.")
        return

    user32 = ctypes.windll.user32

    # Work area excludes the taskbar.
    wa = wt.RECT()
    SPI_GETWORKAREA = 48
    user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0)
    screen_w = wa.right  - wa.left
    screen_h = wa.bottom - wa.top

    # Actual window dimensions (includes title bar + borders) from the first window.
    rect = wt.RECT()
    user32.GetWindowRect(hwnds[0], ctypes.byref(rect))
    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    cols = max(1, screen_w // win_w)
    rows = max(1, screen_h // win_h)
    max_visible = cols * rows

    print(
        f"[window_tiler] {len(hwnds)} windows  {win_w}×{win_h} px each  "
        f"work area {screen_w}×{screen_h}  →  {cols}×{rows} grid "
        f"({max_visible} fully visible)"
    )

    HWND_TOP       = 0
    SWP_NOACTIVATE = 0x0010

    for i, hwnd in enumerate(hwnds):
        if i >= max_visible:
            break
        col = i % cols
        row = i // cols
        x = wa.left + col * win_w
        y = wa.top  + row * win_h
        user32.SetWindowPos(hwnd, HWND_TOP, x, y, win_w, win_h, SWP_NOACTIVATE)


# ── internal helpers ──────────────────────────────────────────────────────────

def _find_windows_by_title(substr: str) -> List[int]:
    """Return HWNDs of all visible top-level windows whose title contains substr."""
    user32  = ctypes.windll.user32
    results: List[int] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def _cb(hwnd: int, _: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if substr in buf.value:
                    results.append(hwnd)
        return True

    # Keep a reference so the callback isn't GC'd mid-enumeration.
    cb = EnumWindowsProc(_cb)
    user32.EnumWindows(cb, 0)
    return results


def _wait_for_windows(
    n: int,
    substr: str,
    timeout: float,
) -> List[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hwnds = _find_windows_by_title(substr)
        if len(hwnds) >= n:
            return hwnds
        time.sleep(0.5)
    # Return however many we found, even if fewer than n.
    return _find_windows_by_title(substr)
