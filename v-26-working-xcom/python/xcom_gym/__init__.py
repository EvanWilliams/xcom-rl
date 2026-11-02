import gymnasium

from .env import XcomInterceptionEnv

gymnasium.register(
    id="XcomInterception-v0",
    entry_point="xcom_gym.env:XcomInterceptionEnv",
    nondeterministic=True,
)

__all__ = ["XcomInterceptionEnv"]
