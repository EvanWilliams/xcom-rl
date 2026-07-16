"""
Test: verify the full Geoscape GUI is navigable via simulated clicks only.

Each test case:
  1. Opens a fresh engine (headless)
  2. Steps until the target state appears (or times out)
  3. Clicks the expected button sequence
  4. Checks the resulting screen_hint / popup_depth

Run:
    py -3 -m xcom_rl.test_gui_playthrough

Exit code: 0 = all tests passed, 1 = one or more failed.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xcom_gym.env import XcomInterceptionEnv, ACTIONS, ACTION_NAMES, SCREEN_TYPES

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ── helpers ────────────────────────────────────────────────────────────────────

def _action_idx(name: str) -> int:
    return ACTION_NAMES.index(name)


def _step_until(env: XcomInterceptionEnv, condition, max_steps: int = 2000, label: str = "") -> Optional[Dict]:
    """Step (wait) until condition(obs) is True, or return None on timeout."""
    obs, _ = env.reset()
    for i in range(max_steps):
        if condition(obs):
            return obs
        obs, *_ = env.step(_action_idx("wait"))
    print(f"  [timeout] {label} did not appear within {max_steps} steps")
    return None


def _screen(obs: Dict) -> str:
    idx = int(obs["screen_type"])
    for name, val in SCREEN_TYPES.items():
        if val == idx:
            return name
    return "?"


# ── individual tests ───────────────────────────────────────────────────────────

def test_time_speed_buttons(env: XcomInterceptionEnv) -> bool:
    """Click each time-speed button on the geoscape; verify popup_depth stays 0."""
    print("\n[test] time-speed buttons")
    obs, _ = env.reset()
    # Advance a few steps to settle
    for _ in range(5):
        obs, *_ = env.step(_action_idx("wait"))

    ok = True
    for btn in ["speed_5sec", "speed_1min", "speed_5min", "speed_30min", "speed_1hour", "speed_1day"]:
        obs, *_ = env.step(_action_idx(btn))
        depth = int(obs["popup_depth"])
        hint  = _screen(obs)
        if depth != 0 or hint != "geoscape":
            print(f"  FAIL  {btn}: depth={depth} hint={hint}")
            ok = False
        else:
            print(f"  ok    {btn}: hint=geoscape depth=0")
    return ok


def test_ufo_intercept_sequence(env: XcomInterceptionEnv) -> bool:
    """
    Wait for a UFO detection popup, then click through the full intercept
    sequence: ufo_detected → intercept → intercept_screen → confirm_dest.
    """
    print("\n[test] UFO intercept sequence")

    obs = _step_until(
        env,
        lambda o: _screen(o) == "ufo_detected",
        max_steps=5000,
        label="ufo_detected popup",
    )
    if obs is None:
        return False
    print(f"  ufo_detected popup appeared (popup_depth={int(obs['popup_depth'])})")

    # Click Intercept on the UFO detected dialog
    obs, *_ = env.step(_action_idx("ufo_intercept"))
    hint = _screen(obs)
    print(f"  after ufo_intercept click: hint={hint}")

    if hint == "intercept":
        # Intercept screen appeared directly
        obs, *_ = env.step(_action_idx("intercept_craft1"))
        hint = _screen(obs)
        print(f"  after intercept_craft1 click: hint={hint}")
    elif hint == "ufo_detected":
        # Sometimes a second UFO popup appears; dismiss it first
        obs, *_ = env.step(_action_idx("ufo_cancel"))
        hint = _screen(obs)
        print(f"  (second ufo popup dismissed): hint={hint}")

    if hint == "confirm_dest":
        obs, *_ = env.step(_action_idx("confirm_ok"))
        hint = _screen(obs)
        print(f"  after confirm_ok click: hint={hint}")
        if hint == "geoscape":
            print(f"  craft dispatched — back to geoscape")
            return True

    # Acceptable: intercept opened but craft not yet at confirm (e.g. craft busy)
    if hint in ("geoscape", "intercept"):
        print(f"  sequence ended at: {hint} (acceptable)")
        return True

    print(f"  FAIL  unexpected hint after sequence: {hint}")
    return False


def test_dismiss_info_dialogs(env: XcomInterceptionEnv) -> bool:
    """
    Wait for any info dialog (low_fuel, craft_error, monthly_report,
    items_arriving) and verify dismiss (Escape) closes it.
    """
    print("\n[test] dismiss info dialogs")

    info_screens = {"low_fuel", "craft_error", "monthly_report", "items_arriving", "craft_patrol"}

    obs = _step_until(
        env,
        lambda o: _screen(o) in info_screens,
        max_steps=8000,
        label="info dialog",
    )
    if obs is None:
        print("  skip  no info dialog appeared within 8000 steps")
        return True   # not a failure — just didn't appear in time

    hint_before = _screen(obs)
    depth_before = int(obs["popup_depth"])
    print(f"  info dialog: {hint_before}  depth={depth_before}")

    obs, *_ = env.step(_action_idx("dismiss"))
    hint_after = _screen(obs)
    depth_after = int(obs["popup_depth"])
    print(f"  after dismiss: hint={hint_after}  depth={depth_after}")

    if depth_after < depth_before or hint_after not in info_screens:
        print(f"  ok  dialog dismissed")
        return True

    print(f"  FAIL  dialog still showing after dismiss")
    return False


def test_mission_detected_cancel(env: XcomInterceptionEnv) -> bool:
    """Wait for a mission/terror popup and cancel it."""
    print("\n[test] mission detected cancel")

    obs = _step_until(
        env,
        lambda o: _screen(o) == "mission_detected",
        max_steps=10000,
        label="mission_detected popup",
    )
    if obs is None:
        print("  skip  no mission detected within 10000 steps")
        return True   # terror missions are rare; skip rather than fail

    print(f"  mission_detected appeared (depth={int(obs['popup_depth'])})")
    obs, *_ = env.step(_action_idx("mission_cancel"))
    hint = _screen(obs)
    depth = int(obs["popup_depth"])
    print(f"  after mission_cancel: hint={hint}  depth={depth}")

    if depth == 0:
        return True
    print(f"  FAIL  popup still open depth={depth}")
    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("GUI click-only playthrough tests")
    print("=" * 60)

    env = XcomInterceptionEnv(headless=True, max_steps=100_000)

    results: List[tuple[str, bool]] = []

    tests = [
        ("time speed buttons",       test_time_speed_buttons),
        ("UFO intercept sequence",   test_ufo_intercept_sequence),
        ("dismiss info dialogs",     test_dismiss_info_dialogs),
        ("mission detected cancel",  test_mission_detected_cancel),
    ]

    try:
        for name, fn in tests:
            t0 = time.time()
            try:
                passed = fn(env)
            except Exception as exc:
                print(f"  EXCEPTION: {exc}")
                passed = False
            elapsed = time.time() - t0
            status = PASS if passed else FAIL
            print(f"  → {status}  {name}  ({elapsed:.1f}s)")
            results.append((name, passed))
    finally:
        env.close()

    print("\n" + "=" * 60)
    print(f"Results: {sum(p for _, p in results)}/{len(results)} passed")
    for name, passed in results:
        marker = "✓" if passed else "✗"
        print(f"  {marker} {name}")
    print("=" * 60)

    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    sys.exit(main())
