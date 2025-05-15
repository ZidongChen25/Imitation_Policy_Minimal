import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

# 1. Reload expert
model = PPO.load("ppo_pendulum_expert")
env = gym.make("Pendulum-v1")

# 2. Rollout expert and collect (obs, action)
observations = []
actions = []

n_episodes = 300
max_steps = 200

for ep in range(n_episodes):
    obs, _ = env.reset()
    for step in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

observations = np.array(observations)
actions = np.array(actions)

# 计算均值和方差
obs_mean = observations.mean(axis=0)
obs_std = observations.std(axis=0) + 1e-8
action_mean = actions.mean(axis=0)
action_std = actions.std(axis=0) + 1e-8



print(f"✅ Collected {len(observations)} (obs, action) pairs.")
print(f"Observations shape: {observations.shape}")
print(f"Actions shape: {actions.shape}")

# 保存
np.savez("expert_demo.npz",
         observations=observations,
         actions=actions,
         obs_mean=obs_mean,
         obs_std=obs_std,
         action_mean=action_mean,
         action_std=action_std)
print("✅ Saved demonstrations to expert_demo.npz")
