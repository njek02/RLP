from stable_baselines3 import SAC
from typing import Callable
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------
# Evaluate success rate
# -----------------------------
def evaluate_success(
    model: SAC,
    make_env_fn: Callable[[], gym.Env],
    episodes: int = 10
) -> float:
    successes = 0
    for _ in range(episodes):
        env = make_env_fn()
        obs, _ = env.reset()
        episode_success = 0
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if info.get("success", 0) == 1:
                episode_success = 1
            if done or truncated:
                break
        successes += episode_success
    return successes / episodes


# -----------------------------
# Success evaluation callback
# -----------------------------
class SuccessEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn: Callable[[], gym.Env],
        eval_freq: int = 50_000,
        episodes: int = 5,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.episodes = episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            success_rate = evaluate_success(
                self.model, self.eval_env_fn, self.episodes
            )
            print(
                f"\n=== Success Rate at {self.num_timesteps} steps: "
                f"{success_rate:.2f} ===\n"
            )
        return True
