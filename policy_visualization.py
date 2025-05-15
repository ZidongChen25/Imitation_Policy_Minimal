from stable_baselines3 import PPO
import gymnasium as gym

# 1. 加载PPO专家模型
model = PPO.load("ppo_pendulum_expert")

# 2. 创建Pendulum环境
env = gym.make("Pendulum-v1", render_mode="human")

# 3. rollout 可视化
for _ in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
