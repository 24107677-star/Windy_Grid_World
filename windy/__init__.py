from gymnasium.envs.registration import register

from .env import WindyGridWorldEnv, load_config

# Cho phép gym.make("WindyGridWorld-v0", config=cfg) giống mẫu GridWorld chính thức
register(id="WindyGridWorld-v0", entry_point="windy.env:WindyGridWorldEnv")

__all__ = ["WindyGridWorldEnv", "load_config"]