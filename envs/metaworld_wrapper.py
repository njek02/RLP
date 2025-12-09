from __future__ import annotations

import metaworld
import gymnasium as gym
import numpy as np
import random

from gymnasium.core import ActType, RenderFrame
from typing import Any, Dict, Optional, Tuple
from stable_baselines3.common.monitor import Monitor
from utils.flatten import flatten_obs


# -----------------------------
# Create a single-task MetaWorld environment
# -----------------------------
def make_env(task_name="reach-v2"):
    ml1 = metaworld.ML1(task_name)

    def _init():
        env_cls = ml1.train_classes[task_name]
        env = env_cls(render_mode=None)
        return MetaWorldWrapper(env, ml1)

    return Monitor(_init())


# -----------------------------
# Custom wrapper for MetaWorld ML1 tasks
# -----------------------------
class MetaWorldWrapper(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, env: gym.Env, ml1: metaworld.ML1):
        super().__init__()
        self.env = env
        self.ml1 = ml1             # store ML1 object
        self.tasks = ml1.train_tasks

        # Set an initial random task
        self._set_random_task()

        # Configure action + obs space
        self.action_space = self.env.action_space

        obs, _ = self.env.reset()
        flat = flatten_obs(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32
        )

    def _set_random_task(self):
        task = random.choice(self.tasks)
        self.env.set_task(task)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Sample new task for each episode
        self._set_random_task()

        obs, info = self.env.reset()
        return flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()
