from __future__ import annotations

import metaworld
import gymnasium as gym
import numpy as np

from gymnasium.core import ActType, RenderFrame
from typing import Any, Dict, Optional, Tuple
from stable_baselines3.common.monitor import Monitor
from utils.flatten import flatten_obs


# -----------------------------
# Create a single-task MetaWorld environment
# -----------------------------
def make_env(task_name: str = "reach-v2") -> gym.Env:
    ml1 = metaworld.ML1(task_name)
    env_cls = ml1.train_classes[task_name]
    env: gym.Env = env_cls()
    task = ml1.train_tasks[0]
    wrapped = MetaWorldWrapper(env, task)
    wrapped = Monitor(wrapped)
    return wrapped


# -----------------------------
# Custom wrapper for MetaWorld ML1 tasks
# -----------------------------
class MetaWorldWrapper(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, ml1_env: gym.Env, task: Any) -> None:
        super().__init__()
        self.env = ml1_env
        self.env.set_task(task)
        self.action_space = self.env.action_space

        obs, _ = self.env.reset()
        flat_obs = flatten_obs(obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        return flatten_obs(obs), info

    def step(
        self,
        action: ActType
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return flatten_obs(obs), reward, terminated, truncated, info

    def render(self) -> Optional[RenderFrame]:
        return self.env.render()
