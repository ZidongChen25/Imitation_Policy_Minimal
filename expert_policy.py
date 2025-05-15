from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("Pendulum-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_pendulum_expert")
print("✅ Expert policy trained and saved.")