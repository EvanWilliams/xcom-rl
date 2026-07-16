"""
Run gymnasium.utils.env_checker.check_env on XcomInterceptionEnv.

Usage (from python/ dir):
    py -3 -m xcom_gym.check_env

The engine is started automatically inside env.reset().
"""
from __future__ import annotations
import sys

from gymnasium.utils.env_checker import check_env
import xcom_gym  # triggers gymnasium.register
from xcom_gym.env import XcomInterceptionEnv


def main() -> None:
    print("=== XcomInterceptionEnv check_env ===\n")

    # Small max_steps so the checker doesn't run full-length episodes
    env = XcomInterceptionEnv(max_steps=500)

    try:
        print("Running gymnasium.utils.env_checker.check_env ...")
        check_env(env, skip_render_check=True)
        print("\n[OK] check_env passed with no errors.")
    except Exception as exc:
        print(f"\n[FAIL] check_env raised: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
