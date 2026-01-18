from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, ProgressBarCallback

from callbacks.success_rate_callback import SuccessEvalCallback
from envs.metaworld_wrapper import make_env
from eval.evaluate_model import evaluate_final_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch ≥ 2.0


SEED = 42

def randomize_masks_globally(saved_masks, sparsity, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Flatten all masks
    flat_masks = []
    shapes = {}

    for key, mask in saved_masks.items():
        flat_masks.append(mask.flatten())
        shapes[key] = mask.shape

    flat = torch.cat(flat_masks)
    total = flat.numel()
    num_zeros = int(total * sparsity)

    # Create random flat mask
    rand_flat = torch.ones(total)
    zero_idx = torch.randperm(total)[:num_zeros]
    rand_flat[zero_idx] = 0.0

    # Split back into layer-shaped masks
    random_masks = {}
    idx = 0
    for key, shape in shapes.items():
        n = torch.prod(torch.tensor(shape)).item()
        random_masks[key] = rand_flat[idx:idx+n].view(shape)
        idx += n

    return random_masks


def _linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _infer_sac_actor_dims(model: SAC) -> List[int]:
    latent_linears = _linear_layers(model.policy.actor.latent_pi)
    dims: List[int] = []
    if latent_linears:
        dims.append(latent_linears[0].in_features)
        dims.extend([layer.out_features for layer in latent_linears])
    else:
        dims.append(model.policy.actor.mu.in_features)
    dims.append(model.policy.actor.mu.out_features)
    return dims


def erk_sparsity(dims: List[int], target_sparsity: float) -> List[float]:
    parameters_num = np.array([dims[i] * dims[i + 1] for i in range(len(dims) - 1)], dtype=np.float64)
    erk_coeffs = np.array(
        [1 - (dims[i] + dims[i + 1]) / parameters_num[i] for i in range(len(dims) - 1)],
        dtype=np.float64,
    )
    k = np.sum(parameters_num) * target_sparsity / np.sum(parameters_num * erk_coeffs)
    return (erk_coeffs * k).tolist()


def build_sac_pruning_schedule(
    model: SAC,
    total_steps: int,
    pruning_start: float,
    pruning_end: float,
    pruning_iterations: int,
    target_sparsity: float,
    use_erk: bool = False,
) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    if pruning_iterations < 1:
        raise ValueError("pruning_iterations must be >= 1")
    if not (0.0 <= pruning_start < pruning_end <= 1.0):
        raise ValueError("pruning_start/end must be within [0,1] and start < end")

    pruning_freq = int(total_steps * (pruning_end - pruning_start) / pruning_iterations)
    if pruning_freq <= 0:
        raise ValueError("pruning_freq must be > 0 (increase total_steps or pruning window)")

    pruning_steps = [
        int(total_steps * pruning_start) + pruning_freq * step
        for step in range(pruning_iterations)
    ]

    dims = _infer_sac_actor_dims(model)

    sparsity_schedule: List[List[float]] = []
    common_sparsities: List[float] = []
    for i in range(pruning_iterations + 1):
        sparsity_point = target_sparsity - target_sparsity * (1 - i / pruning_iterations) ** 3
        common_sparsities.append(sparsity_point)
        if use_erk:
            module_sparsities = erk_sparsity(dims, sparsity_point)
        else:
            module_sparsities = [sparsity_point for _ in range(len(dims) - 1)]
        sparsity_schedule.append(module_sparsities)

    pruning_sparsity_schedule = {step: sp for step, sp in zip(pruning_steps, sparsity_schedule[1:])}
    common_sparsities_schedule = {step: sp for step, sp in zip(pruning_steps, common_sparsities[1:])}
    return pruning_sparsity_schedule, common_sparsities_schedule


def prune_mlp(model: nn.Module, sparsities: List[float]) -> None:
    j = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_mask"):
                zeros = torch.sum(module.weight_mask == 0)
            else:
                zeros = torch.tensor(0, device=module.weight.device)

            total = module.weight.numel()
            current_sparsity = zeros / total

            sparsity_setting = (sparsities[j] - current_sparsity) / (1 - current_sparsity)
            sparsity_setting = sparsity_setting.item()

            if sparsity_setting < 0.0:
                j += 1
                continue

            prune.l1_unstructured(module, name="weight", amount=sparsity_setting)
            module.weight_orig.data = module.weight_orig * module.weight_mask

            if module.bias is not None:
                prune.l1_unstructured(module, name="bias", amount=0.0)
                module.bias_orig.data = module.bias_orig * module.bias_mask

            j += 1


def prune_mlp_remove_parametrization(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, name="weight")
            if hasattr(module, "bias_orig"):
                prune.remove(module, name="bias")

def extract_actor_masks(actor: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract weight masks from all pruned Linear layers in the actor.
    """
    masks = {}
    for name, module in actor.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            masks[f"{name}.weight"] = module.weight_mask.detach().cpu().clone()
    return masks



class SACPruningCallback(BaseCallback):
    def __init__(
        self,
        pruning_sparsity_schedule: Dict[int, List[float]],
        common_sparsities_schedule: Dict[int, float],
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.pruning_schedule = pruning_sparsity_schedule
        self.common_sparsities_schedule = common_sparsities_schedule
        self.last_sparsity = 0.0
        self.pruning_end_step = max(self.pruning_schedule.keys())
        self.finished = False

    def _on_step(self) -> bool:
        step = self.num_timesteps
        self.logger.record("common_sparsity", self.last_sparsity)

        if step > 0 and step in self.pruning_schedule:
            self.last_sparsity = self.common_sparsities_schedule[step]

            prune_mlp(self.model.policy.actor.latent_pi, self.pruning_schedule[step][:-1])
            prune_mlp(self.model.policy.actor.mu, self.pruning_schedule[step][-1:])
            if isinstance(self.model.policy.actor.log_std, nn.Linear):
                prune_mlp(self.model.policy.actor.log_std, self.pruning_schedule[step][-1:])

        if not self.finished and step > self.pruning_end_step:
            self.finished = True

        return True

    def _on_training_end(self) -> None:
        # --------------------------------------------------
        # 1. Save pruning masks BEFORE removing them
        # --------------------------------------------------
        actor = self.model.policy.actor
        masks = extract_actor_masks(actor)

        torch.save(
            masks,
            "actor_pruning_masks.pt",
        )

        if self.verbose > 0:
            print(
                f"[Pruning] Saved {len(masks)} pruning masks to actor_pruning_masks.pt"
            )

        # --------------------------------------------------
        # 2. Remove pruning reparameterization (finalize)
        # --------------------------------------------------
        prune_mlp_remove_parametrization(actor.latent_pi)
        prune_mlp_remove_parametrization(actor.mu)

        if isinstance(actor.log_std, nn.Linear):
            prune_mlp_remove_parametrization(actor.log_std)



# -----------------------------
# Main training
# -----------------------------
def train_model(
    prev_model: Optional[str] = None,
    env_name: str = "peg-insert-side-v3",
    device: str = "cuda",
    buffer: Optional[str] = None,
    total_steps: int = 2_000_000,
    prune_model: bool = False,
    pruning_start: float = 0.6,
    pruning_end: float = 0.9,
    pruning_iterations: int = 3,
    target_sparsity: float = 0.7,
    use_erk: bool = False,
    weights: str = None,
    mask: str = None,
    pretrained: bool = False,
) -> None:
    env = gym.make("Meta-World/MT1", env_name=env_name)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path="./checkpoints/peg_insert_side",
        name_prefix="sac_metaworld_peg_insert",
        save_replay_buffer=True,
    )

    success_callback = SuccessEvalCallback(
        eval_env_fn=lambda: gym.make("Meta-World/MT1", env_name=env_name),
        eval_freq=100_000,
        episodes=10,
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
            seed=SEED,
        )
        timesteps = total_steps

        if weights is not None:
            state_dict = torch.load(weights)

            if pretrained:
                for name, param in model.policy.actor.features_extractor.named_parameters():
                    param.data.copy_(pretrained[f"actor.features_extractor.{name}"])
            else:
                model.policy.load_state_dict(state_dict)
        else:
            torch.save(
                model.policy.state_dict(),
                "initial_policy_weights.pt",
            )
    else:
        print("Loading Model...")
        # Keep learned weights when switching tasks; do not reset to init.
        model = SAC.load(prev_model, env=env, device=device)
        if buffer:
            model.load_replay_buffer(buffer)
        timesteps = max(0, total_steps - model.num_timesteps)

    if mask is not None:
        # Loading masks

        # Apply pruning once to load in weight_mask
        for module in model.actor.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=0.0)

        
        # Load the masks
        print("=" * 10, "Loading masks", "=" * 10)
        masks = torch.load(mask)


        # Apply the masks to the model
        print("=" * 10, "Applying masks", "=" * 10)
        for name, module in model.actor.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
                key = f"{name}.weight"
                if key in masks:
                    print(key)
                    module.weight_mask.data.copy_(
                        masks[key].to(module.weight_mask.device)
                    )
        
        for module in model.actor.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
                module.weight_mask.requires_grad = False


    # callbacks = [checkpoint_callback, success_callback, ProgressBarCallback()]
    callbacks = [ProgressBarCallback()]

    if prune_model:
        pruning_schedule, common_schedule = build_sac_pruning_schedule(
            model=model,
            total_steps=total_steps,
            pruning_start=pruning_start,
            pruning_end=pruning_end,
            pruning_iterations=pruning_iterations,
            target_sparsity=target_sparsity,
            use_erk=use_erk,
        )
        callbacks.append(SACPruningCallback(pruning_schedule, common_schedule))

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    if mask is not None:
        prune_mlp_remove_parametrization(model.actor)


    model.save("sac_metaworld_final")
    print("Training complete! Model saved as sac_metaworld_final.zip")

    # Evaluate model

    evaluate_final_model(
        model,
        lambda: gym.make("Meta-World/MT1", env_name=env_name),
        episodes=100,
        horizon=500,
    )



if __name__ == "__main__":
   
    # Model Training

    train_model(prev_model=None,
                env_name="hand-insert-v3",
                device="cuda",
                prune_model=False,
                weights = None,
                mask = "results/masks/pruned_masks_actor.pt")


    # env = gym.make("Meta-World/MT1", env_name="hand-insert-v3")
    # model = SAC.load("results/models/hand_insert_masked_initial_weights.zip",
    #                   env = env,
    #                   device="cuda")
