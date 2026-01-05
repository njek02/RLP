from __future__ import annotations

from typing import Dict, List, Optional
import copy

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

SEED = 42

def prune_actor_global(actor: nn.Module, amount: float) -> None:
    """
    Globally prune a fraction of remaining weights in the actor.
    """
    parameters_to_prune = []

    for module in actor.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def remove_pruning(actor: nn.Module) -> None:
    for module in actor.modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")


def iterative_pruning_training(
    env_name: str = "peg-insert-side-v3",
    device: str = "cuda",
    total_steps: int = 2_000_000,
    pruning_iterations: int = 5,
    target_sparsity: float = 0.9,
    rewind_steps: int = 100_000,
) -> None:
    """
    Classical IMP:
    - Train dense
    - Prune
    - Rewind weights
    - Repeat
    """

    env = gym.make("Meta-World/MT1", env_name=env_name)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # -------------------------
    # Initial dense model
    # -------------------------
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
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        seed=SEED,
        verbose=0,
    )

    callbacks = [
        ProgressBarCallback(),
    ]

    # -------------------------
    # Dense warmup
    # -------------------------
    print("Dense warmup...")
    model.learn(
        total_timesteps=rewind_steps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    # Save rewind point
    rewind_actor_state = copy.deepcopy(
        model.policy.actor.state_dict()
    )

    remaining_steps = total_steps - rewind_steps
    steps_per_iter = remaining_steps // pruning_iterations

    cumulative_sparsity = 0.0

    # -------------------------
    # IMP iterations
    # -------------------------
    for iteration in range(pruning_iterations):
        print(f"\n=== IMP Iteration {iteration + 1}/{pruning_iterations} ===")

        # Compute per-iteration pruning amount
        target_iter_sparsity = target_sparsity * (iteration + 1) / pruning_iterations
        prune_amount = (
            target_iter_sparsity - cumulative_sparsity
        ) / (1.0 - cumulative_sparsity)

        cumulative_sparsity = target_iter_sparsity

        print(f"Pruning additional {prune_amount:.4f}")

        # Apply pruning
        prune_actor_global(model.policy.actor, prune_amount)

        # Rewind weights
        model.policy.actor.load_state_dict(rewind_actor_state)

        # Train with fixed mask
        model.learn(
            total_timesteps=steps_per_iter,
            callback=callbacks,
            reset_num_timesteps=False,
        )

    # -------------------------
    # Finalize pruning
    # -------------------------
    remove_pruning(model.policy.actor)

    model.save("sac_metaworld_iterative_pruned")
    print("Iterative pruning complete. Model saved.")

if __name__ == "__main__":
    iterative_pruning_training(
        env_name="peg-insert-side-v3",
        device="cuda",
        total_steps=2_000_000,
        pruning_iterations=5,
        target_sparsity=0.9,
        rewind_steps=100_000,
    )
