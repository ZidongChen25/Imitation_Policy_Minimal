import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from gymnasium.vector import AsyncVectorEnv

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)
H = 1  # horizon
T = 10  # flow steps


class NoiseToAction(nn.Module):
    def __init__(self, action_dim, horizon=1, hidden_dim=512):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(action_dim * horizon, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * horizon)
        )

    def forward(self, noise):
        return self.mlp(noise)


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


def flow_matching_loss(policy, obs, action_seq_target, device, pretrain_model):
    batch_size = action_seq_target.shape[0]
    t = torch.rand(batch_size, 1, device=device)
    noise = pretrain_model(torch.randn_like(action_seq_target))
    action_seq_noisy = action_seq_target * t + noise * (1 - t)
    pred_v = policy(obs, action_seq_noisy, t)
    v_target = action_seq_target - noise
    return ((pred_v - v_target) ** 2).mean()


def pretrain():
    # Create log directory
    log_dir = "./logs/pretrain"
    os.makedirs(log_dir, exist_ok=True)

    data = np.load("expert_demo.npz")
    actions = data['actions']
    obs_mean = data['obs_mean']
    obs_std = data['obs_std']
    action_mean = data['action_mean']
    action_std = data['action_std']

    env = gym.make("Pendulum-v1")
    action_dim = env.action_space.shape[0]
    env.close()

    # Build demos
    N = len(actions) - H
    num_train = N // 500
    demos = []
    for i in range(num_train):
        seq = (actions[i:i+H] - action_mean) / action_std
        demos.append(seq.reshape(-1))

    # Initialize model
    noise_model = NoiseToAction(action_dim, horizon=H).to(device)
    optimizer = torch.optim.Adam(noise_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    fixed_noise = torch.randn(num_train, action_dim * H, device=device)
    pretrain_epochs = 10000
    batch_size = 128

    # Training loop
    for epoch in range(pretrain_epochs):
        idx = np.random.choice(num_train, batch_size)
        target = torch.tensor(np.stack([demos[i] for i in idx]), dtype=torch.float32, device=device)
        inp = fixed_noise[idx]
        pred = noise_model(inp)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Pretrain Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(noise_model.state_dict(), os.path.join(log_dir, "pretrain_model.pth"))
    print("✅ Pretrain completed and model saved.")

    # Plot distribution of noise model outputs
    with torch.no_grad():
        samples = torch.randn(10000, action_dim * H, device=device)
        outs = noise_model(samples).cpu().numpy().flatten()
    plt.figure()
    plt.hist(outs, bins=50)
    plt.title('Distribution of Pretrain(noise) Outputs')
    plt.xlim(-2,2)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(log_dir, 'noise_distribution.png'))
    plt.close()
    print(f"📈 Noise distribution plot saved to {log_dir}/noise_distribution.png")


def train_flow():
    # Create log directory
    log_dir = "./logs/flow_matching"
    os.makedirs(log_dir, exist_ok=True)

    data = np.load("expert_demo.npz")
    observations = data['observations']
    actions = data['actions']
    obs_mean = data['obs_mean']
    obs_std = data['obs_std']
    action_mean = data['action_mean']
    action_std = data['action_std']

    N = len(observations) - H
    num_train = N // 500
    demos = []
    for i in range(num_train):
        obs_n = (observations[i] - obs_mean) / obs_std
        act_n = (actions[i:i+H] - action_mean) / action_std
        demos.append((obs_n, act_n.reshape(-1)))

    # Load pretrained noise model
    noise_model = NoiseToAction(actions.shape[-1], horizon=H).to(device)
    noise_model.load_state_dict(torch.load(os.path.join("./logs/pretrain", "pretrain_model.pth"), map_location=device))
    noise_model.eval()
    for p in noise_model.parameters():
        p.requires_grad = False

    # Initialize policy
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    writer = SummaryWriter(log_dir=log_dir)

    epochs = 30000
    batch_size = 32

    for epoch in range(epochs):
        idx = np.random.choice(len(demos), batch_size)
        obs_batch = torch.tensor(np.stack([demos[i][0] for i in idx]), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.stack([demos[i][1] for i in idx]), dtype=torch.float32, device=device)
        loss = flow_matching_loss(policy, obs_batch, action_batch, device, noise_model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 1000 == 0:
            print(f"Flow Epoch {epoch}, Loss: {loss.item():.4f}")

    writer.close()
    torch.save(policy.state_dict(), os.path.join(log_dir, "flow_matching_policy.pth"))
    print("✅ Flow matching training completed and policy saved.")


def make_env():
    return lambda: gym.make("Pendulum-v1", render_mode="rgb_array")


def inference(render_mode='rgb_array', num_envs=5, num_iterations=1):
    if render_mode == "rgb_array":
        env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
    else:
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        num_envs = 1

    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    policy.load_state_dict(torch.load("./logs/flow_matching/flow_matching_policy.pth", map_location=device))
    policy.eval()

    noise_model = NoiseToAction(action_dim, horizon=H).to(device)
    noise_model.load_state_dict(torch.load("./logs/pretrain/pretrain_model.pth", map_location=device))
    noise_model.eval()

    data = np.load("expert_demo.npz")
    obs_mean, obs_std = data['obs_mean'], data['obs_std']
    action_mean, action_std = data['action_mean'], data['action_std']

    all_rewards = []
    for it in range(num_iterations):
        rewards = np.zeros(num_envs)
        done = np.zeros(num_envs, dtype=bool)
        obs, _ = env.reset()
        while not done.all():
            obs_norm = (obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            if render_mode != 'rgb_array':
                obs_tensor = obs_tensor.unsqueeze(0)
            with torch.no_grad():
                action_seq = noise_model(torch.randn((num_envs, action_dim * H), device=device))
                for t_step in range(T):
                    t_tensor = torch.full((num_envs, 1), t_step / T, device=device)
                    v = policy(obs_tensor, action_seq, t_tensor)
                    action_seq = action_seq + v * (1.0 / T)
            act_norm = action_seq.view(num_envs, H, action_dim)[:, 0, :].cpu().numpy()
            final = np.clip(act_norm * action_std + action_mean, -2.0, 2.0)
            obs, reward, term, trunc, _ = env.step(final)
            active = ~(done | term | trunc)
            rewards += reward * active
            done |= (term | trunc)
        all_rewards.extend(rewards)
        print(f"Iteration {it+1}/{num_iterations}: avg reward = {np.mean(rewards):.2f}")
    env.close()
    print(f"✅ Overall avg reward: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flow Matching Pipeline: pretrain, train_flow, or inference.")
    parser.add_argument(
        '--mode', choices=['pretrain', 'train_flow', 'inference_rgb_array', 'inference_human'],
        required=True, help="Select the stage to run."
    )
    parser.add_argument('--num_envs', type=int, default=5, help="Number of parallel envs for inference.")
    parser.add_argument('--num_iterations', type=int, default=1, help="Number of inference iterations.")
    args = parser.parse_args()

    if args.mode == 'pretrain':
        pretrain()
    elif args.mode == 'train_flow':
        train_flow()
    elif args.mode == 'inference_rgb_array':
        inference(render_mode='rgb_array', num_envs=args.num_envs, num_iterations=args.num_iterations)
    elif args.mode == 'inference_human':
        inference(render_mode='human', num_envs=args.num_envs, num_iterations=args.num_iterations)
