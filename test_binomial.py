import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt

H = 1  # horizon
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
T = 10  # number of flow steps per update


class VAE(nn.Module):
    def __init__(self, action_dim, horizon=1, hidden_dim=512, latent_dim=64):
        super().__init__()
        self.horizon = horizon
        self.input_dim = action_dim * horizon
        self.latent_dim = latent_dim

        # Encoder: x -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> hidden -> recon_x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.to(device)  # Ensure z is on the correct device
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction + KL divergence
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


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
        # inputs: [encoded_obs, action_seq_flat, t]
        self.input_proj = nn.Linear(hidden_dim + action_dim * horizon + 1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * horizon),
        )

    def forward(self, obs, action_seq_flat, t):
        # obs: [B, obs_dim], action_seq_flat: [B, action_dim*H], t: [B,1]
        h_obs = self.obs_encoder(obs)
        h = torch.cat([h_obs, action_seq_flat, t], dim=-1)
        h = self.input_proj(h)
        return self.mlp(h)


def flow_matching_loss(policy, obs, action_seq_target, device, pretrain_model):
    """
    obs: [B, obs_dim]
    action_seq_target: [B, action_dim*H]
    """
    batch_size = action_seq_target.shape[0]
    # sample t in [0,1]
    t = torch.rand(batch_size, 1, device=device)
    # generate noise via VAE decoder
    z = torch.randn(batch_size, pretrain_model.latent_dim, device=device)
    noise = pretrain_model.decode(z)
    # interpolate between target and noise
    action_seq_noisy = action_seq_target * t + noise * (1 - t)
    pred_v = policy(obs, action_seq_noisy, t)
    v_target = action_seq_target - noise
    return ((pred_v - v_target) ** 2).mean()


def train():
    # --- Environment & model setup ---
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    pretrain_model = VAE(action_dim, horizon=H, hidden_dim=512, latent_dim=64).to(device)

    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

    pretrain_epochs = 30000
    epochs = 300#00
    batch_size = 32

    writer = SummaryWriter(log_dir="./logs/flow_matching_policy")

    # --- Load expert demos & normalization stats ---
    data = np.load("expert_demo.npz")
    observations = data["observations"]
    actions = data["actions"]
    actions_flat = actions.flatten()
    obs_mean = data["obs_mean"]
    obs_std = data["obs_std"]
    action_mean = data["action_mean"]
    action_std = data["action_std"]

    N = len(observations) - H
    num_train = N // 100

    # build (obs, action_seq) pairs
    demos = []
    for i in range(num_train):
        obs_norm = (observations[i] - obs_mean) / obs_std
        action_seq_norm = (actions[i : i + H] - action_mean) / action_std
        demos.append((obs_norm, action_seq_norm))

    # --- VAE Pretrain ---
    print("🚀 Starting VAE pretraining...")
    for epoch in range(pretrain_epochs):
        # sample a batch of action sequences
        indices = np.random.choice(len(demos), batch_size)
        batch_action = [demos[i][1] for i in indices]  # shape [batch_size, H, action_dim]
        x = np.stack(batch_action).reshape(batch_size, action_dim * H)
        x = torch.tensor(x, dtype=torch.float32, device=device)

        recon_x, mu, logvar = pretrain_model(x)
        loss = vae_loss(recon_x, x, mu, logvar) / batch_size

        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        if epoch % 1000 == 0:
            print(f"[Pretrain] Epoch {epoch}, Loss: {loss.item():.4f}")

    # save and freeze VAE
    torch.save(pretrain_model.state_dict(),
               "./logs/flow_matching_policy/pretrain_model.pth")
    pretrain_model.eval()
    for p in pretrain_model.parameters():
        p.requires_grad = False
    print("✅ VAE pretraining completed, starting flow matching...")
    # Generate noise samples
    num_samples = 10000
    with torch.no_grad():
        z = torch.randn(num_samples, pretrain_model.latent_dim)
        noise = pretrain_model.decode(z).cpu().numpy()
        noise_flat = noise.flatten()

    # Plot distribution of expert actions
    plt.figure()
    plt.hist(actions_flat, bins=50)
    plt.title("Expert Action Distribution")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.show()

    # Plot distribution of pretrained noise
    plt.figure()
    plt.hist(noise_flat, bins=50)
    plt.title("Pretrained VAE Noise Distribution")
    plt.xlabel("Noise Value")
    plt.ylabel("Frequency")
    plt.show()
    # --- Flow-Matching Training ---
    for epoch in range(epochs):
        # sample a batch of (obs, action_seq)
        indices = np.random.choice(len(demos), batch_size)
        batch = [demos[i] for i in indices]
        obs_batch, _ = zip(*batch)
        obs_batch = np.stack(obs_batch)
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)

        # initialize action_seq via VAE decoder
        with torch.no_grad():
            z = torch.randn((batch_size, pretrain_model.latent_dim),
                            device=device)
            action_seq = pretrain_model.decode(z)

        # compute flow-matching loss
        loss = flow_matching_loss(policy, obs_batch,
                                  action_seq, device, pretrain_model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 1000 == 0:
            print(f"[Flow] Epoch {epoch}, Loss: {loss.item():.4f}")

    writer.close()
    torch.save(policy.state_dict(),
               "./logs/flow_matching_policy/test.pth")
    print("✅ Flow-matching training completed and model saved.")


def make_env():
    return lambda: gym.make("Pendulum-v1", render_mode="rgb_array")


def inference(render_mode='rgb_array', num_envs=50, num_episodes=10):
    # --- Environment setup ---
    if num_envs > 1:
        envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
        env = envs
    else:
        env = gym.make("Pendulum-v1", render_mode=render_mode)

    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    # --- Load trained policy & VAE ---
    policy = FlowMatchingPolicy(obs_dim, action_dim, horizon=H).to(device)
    policy.load_state_dict(torch.load(
        "./logs/flow_matching_policy/test.pth", map_location=device))
    policy.eval()

    pretrain_model = VAE(action_dim, horizon=H,
                         hidden_dim=512, latent_dim=64).to(device)
    pretrain_model.load_state_dict(torch.load(
        "./logs/flow_matching_policy/pretrain_model.pth", map_location=device))
    pretrain_model.eval()

    # --- Load normalization stats ---
    data = np.load("expert_demo.npz")
    obs_mean, obs_std = data['obs_mean'], data['obs_std']
    action_mean, action_std = data['action_mean'], data['action_std']

    all_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_rewards = np.zeros(num_envs)
        terminated = np.zeros(num_envs, dtype=bool)
        truncated = np.zeros(num_envs, dtype=bool)

        while not (terminated & truncated).all():
            # normalize obs
            obs_norm = (obs - obs_mean) / obs_std
            if render_mode == 'rgb_array' or num_envs > 1:
                obs_tensor = torch.tensor(obs_norm,
                                          dtype=torch.float32,
                                          device=device)
            else:
                obs_tensor = torch.tensor(obs_norm,
                                          dtype=torch.float32,
                                          device=device).unsqueeze(0)

            # sample initial action_seq from VAE
            with torch.no_grad():
                z = torch.randn((num_envs, pretrain_model.latent_dim),
                                device=device)
                action_seq = pretrain_model.decode(z)

            # flow-matching integration
            for t_step in range(T):
                t_tensor = torch.full((num_envs, 1),
                                      t_step / T,
                                      device=device)
                v = policy(obs_tensor, action_seq, t_tensor)
                action_seq = action_seq + v * (1.0 / T)

            # de-normalize and step
            action_flat = action_seq.cpu().numpy()
            action = (action_flat * action_std + action_mean) \
                     .reshape(num_envs, H, action_dim)[:, 0, :]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards += reward

        avg_ret = episode_rewards.mean()
        all_returns.append(avg_ret)
        print(f"Episode {ep} return: {avg_ret:.2f}")

    print("=== Inference finished ===")
    print("Average return over all episodes:",
          np.mean(all_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training or inference.")
    parser.add_argument('--mode',
                        choices=['train',
                                 'inference_rgb_array',
                                 'inference_human'],
                        required=True,
                        help="Specify 'train' to train the model, "
                             "'inference_rgb_array' or 'inference_human' to run inference.")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'inference_rgb_array':
        inference(render_mode='rgb_array',
                  num_envs=50,
                  num_episodes=10)
    elif args.mode == 'inference_human':
        inference(render_mode='human',
                  num_envs=1,
                  num_episodes=1)

