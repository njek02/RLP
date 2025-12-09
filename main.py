from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from callbacks.success_rate_callback import SuccessEvalCallback
from envs.metaworld_wrapper import make_env
from eval.evaluate_model import evaluate_final_model

# -----------------------------
# Main training
# -----------------------------
def train_model(prev_model: str = None, env_name: str = "peg-insert-side-v3", device: str = "cuda") -> None:
    env = make_env(env_name)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/peg_insert_side",
        name_prefix="sac_metaworld_peg_insert",
    )

    success_callback = SuccessEvalCallback(
        eval_env_fn=lambda: make_env(env_name),
        eval_freq=100_000,
        episodes=5
    )

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if prev_model is None:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            batch_size=500,
            gamma=0.99,
            tau=5e-3,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            buffer_size=1_000_000,
            policy_kwargs=dict(
                net_arch=[256, 256],
            ),
            verbose=0,
            device=device,
            seed=seed,
        )

        # Save weight initialization
        model.save("sac_metaworld_peg_insert_init")
        print("Initial weights saved.")

        timesteps = 2_000_000
    else:
        model = SAC.load(prev_model, env=env, device="cuda")
        timesteps = 2_000_000 - model.num_timesteps

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, success_callback, ProgressBarCallback()],
        reset_num_timesteps=False
    )

    model.save("sac_metaworld_final")
    print("Training complete! Model saved as sac_metaworld_final.zip")


if __name__ == "__main__":
    # train_model(prev_model="checkpoints\peg_insert_side\sac_metaworld_peg_insert_1000000_steps.zip", env_name="peg-insert-side-v3", device="cuda")

    # Model Evaluation
    model = SAC.load("models/peg_insert_side/peg_insert_final.zip")
    env_name = "peg-insert-side-v3"

    evaluate_final_model(
        model,
        lambda: make_env(env_name),
        episodes=50,
        horizon=500,
    )
