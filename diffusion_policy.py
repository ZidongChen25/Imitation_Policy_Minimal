import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # <-- 新增Tensorboard
H = 1# multi-step horizon

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon=10, hidden_dim=128, n_layers=2, n_heads=2):
        super().__init__()
        self.horizon = horizon
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(hidden_dim + (action_dim * horizon) + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(hidden_dim, action_dim * horizon)

    def forward(self, obs, noisy_action_seq, timestep):
        obs_feat = self.obs_encoder(obs)
        x = torch.cat([obs_feat, noisy_action_seq, timestep], dim=-1)
        x = self.input_proj(x).unsqueeze(0)  # Add sequence dim
        x = self.transformer(x)
        x = x.squeeze(0)
        return self.output_proj(x)



# 2. Noise injection function
def q_sample(actions, t, noise):
    alpha = 1.0 - (t / T)
    return torch.sqrt(alpha) * actions + torch.sqrt(1 - alpha) * noise

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 50  # diffusion steps

def train():
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = -2.0
    action_high = 2.0

    policy = DiffusionPolicy(obs_dim, action_dim, horizon=H).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    epochs = 30000
    batch_size = 32

    writer = SummaryWriter(log_dir="./logs/diffusion_policy_H5")

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

        t = torch.randint(1, T, (batch_size, 1), device=device)
        noise = torch.randn_like(action_seq_batch)
        noisy_action_seq = q_sample(action_seq_batch, t.float(), noise)

        pred_noise_seq = policy(obs_batch, noisy_action_seq, t.float() / T)
        loss = ((pred_noise_seq - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    writer.close()
    torch.save(policy.state_dict(), "./logs/diffusion_policy_H5/diffusion_policy_H5.pth")
    print("✅ Model saved.")

def inference():
    env = gym.make("Pendulum-v1",render_mode='human')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = -2.0
    action_high = 2.0

    policy = DiffusionPolicy(obs_dim, action_dim, horizon=H).to(device)
    policy.load_state_dict(torch.load("./logs/diffusion_policy_H5/diffusion_policy_H5.pth", map_location=device))
    policy.eval()

    data = np.load("expert_demo.npz")
    obs_mean = data['obs_mean']
    obs_std = data['obs_std']
    action_mean = data['action_mean']
    action_std = data['action_std']

    episode_rewards = []

    for _ in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            env.render()
            obs_norm = (obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            action_seq = torch.randn((1, action_dim * H), device=device)
            for t_step in reversed(range(1, T + 1)):
                t_tensor = torch.full((1, 1), t_step / T, device=device)
                noise_pred = policy(obs_tensor, action_seq, t_tensor)
                action_seq = (action_seq - (1.0 / T) * noise_pred)
            action_norm = action_seq.view(1, H, action_dim)[:, 0, :].squeeze(0).cpu().detach().numpy()
            final_action = action_norm * action_std + action_mean
            final_action = np.clip(final_action, action_low, action_high)
            obs, reward, terminated, truncated, info = env.step(final_action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    env.close()
    average_reward = np.mean(episode_rewards)
    print(f"✅ Average reward over {len(episode_rewards)} episodes: {average_reward:.2f}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or inference.")
    parser.add_argument('--mode', choices=['train', 'inference'], required=True, 
                        help="Specify 'train' to train the model or 'inference' to run inference.")
    args = parser.parse_args()

    if args.mode == 'train':
        train()  # 只运行训练
    elif args.mode == 'inference':
        inference()  # 只运行推理
