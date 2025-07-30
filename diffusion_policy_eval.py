import argparse
import math
import os
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  # registers Push‑T envs
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------
# Sinusoidal Timestep Embedding (from DDPM baseline)
# --------------------------------------------------
class SinusoidalEmbedding(nn.Module):
    """Create sinusoidal embeddings of timesteps (dimension = hidden_dim).

    This is identical to the implementation used by Ho et al. (2020) and
    reused by the original Diffusion‑Policy codebase."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """timesteps: (batch, 1) or (batch,) tensor of ints / floats"""
        if timesteps.dim() == 2 and timesteps.size(1) == 1:
            timesteps = timesteps.squeeze(1)
        half_dim = self.hidden_dim // 2
        exponent = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        angles = timesteps.float() * torch.exp(-math.log(10000.0) * exponent)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # (batch, hidden_dim)


# --------------------------------------------------
# Diffusion Policy Network – Original Architecture
# --------------------------------------------------
class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 16,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 1) Observation encoder (MLP → hidden_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2) Action embedding (linear per‑timestep)
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # 3) Learnable positional embedding for each step in horizon
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        # 4) Sinusoidal time embedding → MLP (original DP style)
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # 5) Transformer encoder (identical hyper‑params to original code)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 6) Output projection back to flattened action sequence
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,  # (B, obs_dim)
        noisy_action_seq: torch.Tensor,  # (B, horizon * action_dim)
        timesteps: torch.Tensor,  # (B, 1) integer (1‑T)
    ) -> torch.Tensor:
        B = obs.size(0)

        # --- encode observation & replicate across horizon ---
        obs_token = self.obs_encoder(obs)  # (B, hidden)
        obs_token = obs_token.unsqueeze(1).expand(B, self.horizon, self.hidden_dim)

        # --- embed (noisy) action sequence ---
        act_seq = noisy_action_seq.view(B, self.horizon, self.action_dim)
        act_token = self.action_embed(act_seq)  # (B, horizon, hidden)

        # --- time embedding broadcast ---
        temb = self.time_mlp(timesteps).unsqueeze(1)  # (B, 1, hidden)

        # --- combine tokens (+ positional) ---
        x = obs_token + act_token + self.pos_embed + temb  # (B, horizon, hidden)

        # transformer expects (seq, batch, hidden)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, horizon, hidden)

        # project back to action space and flatten
        pred_noise = self.output_proj(x).reshape(B, -1)  # (B, horizon*action_dim)
        return pred_noise


# --------------------------------------------------
# Diffusion Utilities
# --------------------------------------------------
@torch.no_grad()
def q_sample(actions: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, T: int):
    """Forward diffusion q(x_t | x_0) with linear schedule α_t = 1 - t/T."""
    alpha = 1.0 - (t / T)
    return torch.sqrt(alpha) * actions + torch.sqrt(1 - alpha) * noise


# --------------------------------------------------
# Training loop
# --------------------------------------------------

def train(
    demo_path: str,
    log_dir: str = "logs/pusht",
    horizon: int = 16,
    T: int = 50,
    epochs: int = 50_000,
    batch_size: int = 512,
    lr: float = 3e-4,
):
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    obs_dim = env.observation_space.shape[0]  # 5
    action_dim = env.action_space.shape[0]  # 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionPolicy(obs_dim, action_dim, horizon=horizon).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr)
    writer = SummaryWriter(log_dir)

    # ----- load expert demonstrations -----
    data = np.load(demo_path)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)

    obs_mean = data.get("obs_mean", observations.mean(axis=0))
    obs_std = data.get("obs_std", observations.std(axis=0) + 1e-6)
    act_mean = data.get("action_mean", actions.mean(axis=0))
    act_std = data.get("action_std", actions.std(axis=0) + 1e-6)

    np.savez(Path(log_dir) / "normaliser.npz", **{
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": act_mean,
        "action_std": act_std,
    })

    # ----- build (obs, action_seq) windows -----
    demos_obs, demos_act = [], []
    for i in range(len(observations) - horizon):
        demos_obs.append((observations[i] - obs_mean) / obs_std)
        seq = (actions[i : i + horizon] - act_mean) / act_std
        demos_act.append(seq.reshape(-1))
    demos_obs = np.asarray(demos_obs, dtype=np.float32)
    demos_act = np.asarray(demos_act, dtype=np.float32)

    # ----- training -----
    for epoch in range(epochs):
        idx = np.random.randint(0, demos_obs.shape[0], size=batch_size)
        obs_batch = torch.as_tensor(demos_obs[idx], device=device)
        act_batch = torch.as_tensor(demos_act[idx], device=device)

        t_int = torch.randint(1, T, (batch_size, 1), device=device)
        noise = torch.randn_like(act_batch)
        noisy_act = q_sample(act_batch, t_int.float(), noise, T)

        pred_noise = policy(obs_batch, noisy_act, t_int)
        loss = nn.functional.mse_loss(pred_noise, noise)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        writer.add_scalar("train/loss", loss.item(), epoch)
        if epoch % 500 == 0:
            print(f"[epoch {epoch:05d}] loss = {loss.item():.6f}")

    ckpt = Path(log_dir) / "policy.pt"
    torch.save(policy.state_dict(), ckpt)
    writer.close()
    print(f"✅ Training complete. Checkpoint saved to {ckpt}")


# --------------------------------------------------
# Evaluation loop (vectorised)
# --------------------------------------------------
@torch.no_grad()
def evaluate(
    checkpoint: str,
    log_dir: str,
    horizon: int = 16,
    T: int = 50,
    max_episodes: int = 500,
    num_envs: int = 10,
    render_mode: str = "rgb_array",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalisation stats
    stats = np.load(Path(log_dir) / "normaliser.npz")
    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    act_mean, act_std = stats["action_mean"], stats["action_std"]

    # env helper
    def make_env():
        return lambda: gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)

    venv = num_envs > 1 and render_mode == "rgb_array"
    if venv:
        env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    else:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
        num_envs = 1

    obs_dim = env.single_observation_space.shape[0] if venv else env.observation_space.shape[0]
    act_dim = env.single_action_space.shape[0] if venv else env.action_space.shape[0]

    # load policy
    policy = DiffusionPolicy(obs_dim, act_dim, horizon=horizon).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()

    # episode bookkeeping
    success = np.zeros(num_envs)
    done_flags = np.zeros(num_envs, dtype=bool)
    episodes_finished = 0

    obs, _ = env.reset()

    while episodes_finished < max_episodes:
        obs_norm = (obs - obs_mean) / obs_std
        obs_tensor = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
        if not venv:
            obs_tensor = obs_tensor.unsqueeze(0)

        # reverse diffusion sampling
        act_seq = torch.randn((num_envs, act_dim * horizon), device=device)
        for t_step in reversed(range(1, T + 1)):
            t_tensor = torch.full((num_envs, 1), t_step, device=device)
            noise_pred = policy(obs_tensor, act_seq, t_tensor)
            act_seq -= noise_pred / T

        first_action = act_seq.view(num_envs, horizon, act_dim)[:, 0, :].cpu().numpy()
        final_action = np.clip(first_action * act_std + act_mean,
                               env.single_action_space.low if venv else env.action_space.low,
                               env.single_action_space.high if venv else env.action_space.high)

        obs, _, terminated, truncated, infos = env.step(final_action)

        # update episode termination bookkeeping
        if venv:
            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    success[i] += infos[i].get("is_success", False)
                    episodes_finished += 1
                    obs_i, _ = env.reset_at(i)
                    obs[i] = obs_i
        else:
            if terminated or truncated:
                success[0] += infos.get("is_success", False)
                episodes_finished += 1
                obs, _ = env.reset()

    env.close()
    success_rate = success.sum() / max_episodes
    print(f"✅ Evaluation done: Success‑rate = {success_rate*100:.1f}% over {max_episodes} episodes")


# --------------------------------------------------
# Entry‑point
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--demo_path", type=str, default="demos/pusht_state_expert.npz")
    parser.add_argument("--log_dir", type=str, default="logs/pusht")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt (for evaluation)")
    parser.add_argument("--num_envs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "train":
        train(demo_path=args.demo_path, log_dir=args.log_dir)
    else:
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.log_dir) / "policy.pt")
        evaluate(checkpoint=args.checkpoint, log_dir=args.log_dir, num_envs=args.num_envs)
