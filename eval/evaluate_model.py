from .graphs import plot_success_rate_se

import numpy as np


def evaluate_final_model(model, make_env_fn, episodes=20, horizon=200, runs=10):
    success_per_run = []
    efficiency_per_run = []

    for run in range(runs):
        successes = 0
        rewards = []
        steps_to_success = []

        for ep in range(episodes):
            env = make_env_fn()
            obs, _ = env.reset()

            ep_reward = 0.0
            ep_success = False

            for num_step in range(horizon):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                ep_reward += reward

                if info.get("success", 0) == 1:
                    ep_success = True
                    steps_to_success.append(num_step)
                    break

                if done or truncated:
                    break

            rewards.append(ep_reward)
            successes += int(ep_success)

            env.close()

        success_rate = successes / episodes
        avg_reward = np.mean(rewards)
        avg_efficiency = np.mean(steps_to_success)

        success_per_run.append(success_rate)
        efficiency_per_run.append(avg_efficiency)

        print(f"Run {run + 1}: Success rate: {success_rate:.2f}")
        print(f"Run {run + 1}: Average reward: {avg_reward:.2f}")
        print(f"Run {run + 1}: Average efficiency: {avg_efficiency:.2f} steps")
    
    return success_per_run, efficiency_per_run

    # plot_success_rate_se(success_per_run, True)

