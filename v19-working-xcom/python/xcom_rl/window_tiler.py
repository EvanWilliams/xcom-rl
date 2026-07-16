"""
Tile OpenXcom training windows to maximise fully-visible count.

Pure-ctypes Win32 — no pywin32 dependency.
Safely no-ops on non-Windows platforms.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

_ENV_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def watch_and_retile(
    n: int,
    title_substr: str = "OpenXcom",
    check_interval: float = 2.0,
    log_dir: str = _ENV_LOG_DIR,
) -> None:
    """
    Long-running background thread entry point.

    1. Waits for *n* windows to appear and tiles them (initial layout).
    2. Keeps polling every *check_interval* seconds; whenever a new HWND
       appears (engine crashed and restarted), waits briefly for any other
       restarting windows then retiles the full set.
    3. Updates each window's title bar with its current episode reward and
       rank (highest reward = #1); retiles in rank order when rankings change.

    Runs until the process exits (designed for a daemon thread).
    """
    if sys.platform != "win32":
        return

    _do_tile(n, title_substr, timeout=30.0)
    known: set[int] = set(_find_windows_by_title(title_substr))
    prev_sorted_ranks: list[int] = []

    while True:
        time.sleep(check_interval)
        current: set[int] = set(_find_windows_by_title(title_substr))
        if current - known:
            time.sleep(1.5)
            _do_tile(n, title_substr, timeout=15.0)
            known = set(_find_windows_by_title(title_substr))
            prev_sorted_ranks = []
        else:
            known = current

        rewards  = _read_rewards(n, log_dir)
        rank_hwnd = _rank_to_hwnd(n, log_dir)
        if not rewards or not rank_hwnd:
            continue

        sorted_ranks = sorted(rewards, key=rewards.__getitem__, reverse=True)

        user32 = ctypes.windll.user32
        for pos, rank in enumerate(sorted_ranks):
            hwnd = rank_hwnd.get(rank)
            if hwnd:
                user32.SetWindowTextW(hwnd, f"OpenXcom #{pos + 1}  reward={rewards[rank]:+.1f}")

        if sorted_ranks != prev_sorted_ranks:
            hwnd_order = [rank_hwnd[r] for r in sorted_ranks if r in rank_hwnd]
            if len(hwnd_order) == n:
                _layout_windows(hwnd_order)
            prev_sorted_ranks = sorted_ranks


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
    hwnds = _wait_for_windows(n, title_substr, timeout)
    if not hwnds:
        print("[window_tiler] No OpenXcom windows found within timeout; skipping tile.")
        return
    _layout_windows(hwnds)


def _layout_windows(hwnds: List[int]) -> None:
    """Position *hwnds* in a grid that maximises per-window visible area."""
    if not hwnds:
        return

    user32 = ctypes.windll.user32

    wa = wt.RECT()
    SPI_GETWORKAREA = 48
    user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0)
    screen_w = wa.right  - wa.left
    screen_h = wa.bottom - wa.top

    rect = wt.RECT()
    user32.GetWindowRect(hwnds[0], ctypes.byref(rect))
    win_w = rect.right  - rect.left
    win_h = rect.bottom - rect.top

    n = len(hwnds)
    best_cols, best_rows, best_x_step, best_y_step = 1, n, 0.0, 0.0
    best_score = -1.0
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        x_step = max(0.0, (screen_w - win_w) / (cols - 1)) if cols > 1 else 0.0
        y_step = max(0.0, (screen_h - win_h) / (rows - 1)) if rows > 1 else 0.0
        score = min(x_step, float(win_w)) * min(y_step, float(win_h))
        if cols == 1:
            score = float(win_w) * min(y_step, float(win_h))
        if rows == 1:
            score = min(x_step, float(win_w)) * float(win_h)
        if cols == 1 and rows == 1:
            score = float(win_w * win_h)
        if score > best_score:
            best_score = score
            best_cols, best_rows = cols, rows
            best_x_step, best_y_step = x_step, y_step

    print(
        f"[window_tiler] {n} windows  {win_w}×{win_h} px  "
        f"work area {screen_w}×{screen_h}  →  {best_cols}×{best_rows} "
        f"step ({int(best_x_step)}×{int(best_y_step)} px)"
    )

    HWND_TOP       = 0
    SWP_NOACTIVATE = 0x0010

    for i, hwnd in enumerate(hwnds):
        col = i % best_cols
        row = i // best_cols
        x = wa.left + round(col * best_x_step)
        y = wa.top  + round(row * best_y_step)
        user32.SetWindowPos(hwnd, HWND_TOP, x, y, win_w, win_h, SWP_NOACTIVATE)


# ── reward helpers ────────────────────────────────────────────────────────────

def _read_rewards(n_envs: int, log_dir: str) -> Dict[int, float]:
    """Read ep_reward from live_env_{rank}.json for each rank. Skip missing files."""
    rewards: Dict[int, float] = {}
    for rank in range(n_envs):
        path = os.path.join(log_dir, f"live_env_{rank}.json")
        try:
            with open(path) as f:
                data = json.load(f)
            rewards[rank] = float(data["ep_reward"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    return rewards


def _hwnd_for_pid(pid: int) -> Optional[int]:
    """Return the first visible top-level HWND belonging to *pid*, or None."""
    user32 = ctypes.windll.user32
    result: List[int] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def _cb(hwnd: int, _: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            found_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(found_pid))
            if found_pid.value == pid:
                result.append(hwnd)
        return True

    cb = EnumWindowsProc(_cb)
    user32.EnumWindows(cb, 0)
    return result[0] if result else None


def _rank_to_hwnd(n_envs: int, log_dir: str) -> Dict[int, int]:
    """Map env rank → HWND by reading engine_{rank}.pid files."""
    rank_hwnd: Dict[int, int] = {}
    for rank in range(n_envs):
        pid_path = os.path.join(log_dir, f"engine_{rank}.pid")
        try:
            with open(pid_path) as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            continue
        hwnd = _hwnd_for_pid(pid)
        if hwnd is not None:
            rank_hwnd[rank] = hwnd
    return rank_hwnd


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
    return _find_windows_by_title(substr)
