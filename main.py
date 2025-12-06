from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks.success_rate_callback import SuccessEvalCallback
from envs.metaworld_wrapper import make_env


# -----------------------------
# Main training
# -----------------------------
def main() -> None:
    env = make_env("reach-v3")

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="sac_metaworld"
    )

    success_callback = SuccessEvalCallback(
        eval_env_fn=lambda: make_env("reach-v3"),
        eval_freq=2_000,
        episodes=5
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 256])
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, success_callback]
    )

    model.save("sac_metaworld_final")
    print("Training complete! Model saved as sac_metaworld_final.zip")


if __name__ == "__main__":
    main()
