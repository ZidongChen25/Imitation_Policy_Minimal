import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

H = 1  # horizon

class FlowMatchingPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon=1, hidden_dim=128):
        super().__init__()
        self.horizon = horizon
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(hidden_dim + (action_dim * horizon) + 1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * horizon)
        )

    def forward(self, obs, action_seq, t):
        obs_feat = self.obs_encoder(obs)
        x = torch.cat([obs_feat, action_seq, t], dim=-1)
        x = self.input_proj(x)
        return self.mlp(x)

def flow_matching_loss(policy, obs, action_seq_target, device):
    # action_seq_target: [B, action_dim * H], target action
    # Sample t in [0, 1]
    batch_size = action_seq_target.shape[0]
    t = torch.rand(batch_size, 1, device=device)
    # Sample noisy action between action_seq and random noise
    noise = torch.randn_like(action_seq_target)
    action_seq_noisy = action_seq_target * t + noise * (1 - t)
    pred_v = policy(obs, action_seq_noisy, t)
    v_target = (action_seq_target - noise)  #This is the derivative of (action_seq_noisy-action_seq_target) respect to t
    return ((pred_v - v_target) ** 2).mean()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
T = 10 # flow steps

def train():
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    epochs = 30000
    batch_size = 32

    writer = SummaryWriter(log_dir="./logs/flow_matching_policy")

    data = np.load("expert_demo.npz")
    observations = data['observations']
    actions = data['actions']
    obs_mean = data['obs_mean']
    obs_std = data['obs_std']
    action_mean = data['action_mean']
    action_std = data['action_std']

    demos = []
    for i in range(len(observations) - H):
        obs = (observations[i] - obs_mean) / obs_std
        action_seq = (actions[i:i+H] - action_mean) / action_std
        demos.append((obs, action_seq))

    for epoch in range(epochs):
        indices = np.random.choice(len(demos), batch_size)
        batch = [demos[i] for i in indices]
        obs_batch = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(device)
        action_seq_batch = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).to(device)
        action_seq_batch = action_seq_batch.view(batch_size, -1)

        # Target is the original action_seq
        loss = flow_matching_loss(policy, obs_batch, action_seq_batch,device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    writer.close()
    torch.save(policy.state_dict(), "./logs/flow_matching_policy/flow_matching_policy.pth")
    print("✅ Model saved.")
from gymnasium.vector import AsyncVectorEnv

def make_env():
    return lambda: gym.make("Pendulum-v1", render_mode="rgb_array")

def inference(render_mode='rgb_array', num_envs=5):
    if render_mode == "rgb_array":
        env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
    else:
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        num_envs = 1 

    action_low = -2.0
    action_high = 2.0

    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    policy.load_state_dict(torch.load("./logs/flow_matching_policy/flow_matching_policy.pth", map_location=device))
    policy.eval()

    data = np.load("expert_demo.npz")
    obs_mean = data['obs_mean']
    obs_std = data['obs_std']
    action_mean = data['action_mean']
    action_std = data['action_std']

    episode_rewards = np.zeros(num_envs)
    terminated_flags = np.zeros(num_envs, dtype=bool)

    obs, _ = env.reset()

    while not terminated_flags.all():
        obs_norm = (obs - obs_mean) / obs_std
        if render_mode == 'rgb_array':
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device)
        else:
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

        action_seq = torch.randn((num_envs, action_dim * H), device=device)
        for t_step in range(T):
            t_tensor = torch.full((num_envs, 1), t_step / T, device=device)
            v = policy(obs_tensor, action_seq, t_tensor)
            action_seq = action_seq + v * (1.0 / T)

        action_norm = action_seq.view(num_envs, H, action_dim)[:, 0, :].squeeze(0).cpu().detach().numpy()
        final_action = action_norm * action_std + action_mean
        final_action = np.clip(final_action, action_low, action_high)

        obs, reward, terminated, truncated, info = env.step(final_action)
        active = ~(terminated_flags | terminated | truncated)
        episode_rewards += reward * active
        terminated_flags |= (terminated | truncated)

    env.close()
    print(f"✅ Average reward over {num_envs} parallel episodes: {np.mean(episode_rewards):.2f}")
# def inference(render_mode='rgb_array'):
#     env = gym.make("Pendulum-v1", render_mode=render_mode)
#     obs_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     action_low = -2.0
#     action_high = 2.0

#     policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
#     policy.load_state_dict(torch.load("./logs/flow_matching_policy/flow_matching_policy.pth", map_location=device))
#     policy.eval()

#     data = np.load("expert_demo.npz")
#     obs_mean = data['obs_mean']
#     obs_std = data['obs_std']
#     action_mean = data['action_mean']
#     action_std = data['action_std']

#     episode_rewards = []

#     for _ in range(5):
#         obs, _ = env.reset()
#         done = False
#         total_reward = 0.0

#         while not done:
#             obs_norm = (obs - obs_mean) / obs_std
#             obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
#             # Start from random action, integrate flow
#             action_seq = torch.randn((1, action_dim * H), device=device)
#             steps = T
#             for i in range(steps):
#                 t = torch.full((1, 1), i / steps, device=device)
#                 v = policy(obs_tensor, action_seq, t)
#                 action_seq = action_seq + v * (1.0 / steps)
#             action_norm = action_seq.view(1, H, action_dim)[:, 0, :].squeeze(0).cpu().detach().numpy()
#             final_action = action_norm * action_std + action_mean
#             final_action = np.clip(final_action, action_low, action_high)
#             obs, reward, terminated, truncated, info = env.step(final_action)
#             total_reward += reward
#             done = terminated or truncated
#         episode_rewards.append(total_reward)
#     env.close()
#     average_reward = np.mean(episode_rewards)
#     print(f"✅ Average reward over {len(episode_rewards)} episodes: {average_reward:.2f}")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or inference.")
    parser.add_argument('--mode', choices=['train', 'inference_rgb_array', 'inference_human'], required=True, 
                        help="Specify 'train' to train the model or 'inference_rgb_array' or 'inference_human' to run inference.")
    args = parser.parse_args()

    if args.mode == 'train':
        train()  
    elif args.mode == 'inference_rgb_array':
        inference(render_mode='rgb_array')  
    elif args.mode == 'inference_human':
        inference(render_mode='human')  