import numpy as np

def evaluate_final_model(model, make_env_fn, episodes=20, horizon=200):
    successes = 0
    rewards = []

    for ep in range(episodes):
        env = make_env_fn()
        obs, _ = env.reset()
        ep_reward = 0
        ep_success = False

        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if info.get("success", 0) == 1:
                ep_success = True

            if done or truncated:
                break

        rewards.append(ep_reward)
        successes += int(ep_success)

    success_rate = successes / episodes
    avg_reward = np.mean(rewards)

    return success_rate, avg_reward
