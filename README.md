
# 🤖 Imitation Policy Minimal

Are you tired of constantly switching between code files, papers, and tutorials?  
Frustrated by endless dependency installations and environment conflicts—when all you want is to run simple demos and learn from the code?

**Imitation Policy Minimal** is a clean, educational implementation of imitation learning-based policies for embodied AI.  
This project integrates both **Diffusion Policy** and **Flow Matching** for simple control tasks like `Pendulum-v1`.

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/ZidongChen25/Imitation_Policy_Minimal.git
cd Imitation_Policy_Minimal
```

### Create a Python environment (optional but recommended)

```bash
conda create -n diffusion_policy_minimal python=3.10
conda activate diffusion_policy_minimal
```

### Install required packages

```bash
pip install torch gymnasium tensorboard stable-baselines3 numpy pygame
```
Users planning to leverage GPU acceleration should install the appropriate CUDA-enabled build of PyTorch (e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118) or follow PyTorch’s official installation instructions for their CUDA version. For CPU-only use, the standard PyPI build of torch is sufficient.

## 🚀 How to Use

Imitation learning policy requires expert demonstrations, for example, human demonstration or we can train a RL policy such as PPO, SAC, DDPG. In this easy environment Pendulum, we use PPO to train an expert policy.

1. Generate expert demonstrations:

   ```bash
   python expert_policy.py
   ```

2. Create demonstrations:

   ```bash
   python generate_demonstration.py
   ```

   This will create an `expert_demo.npz` file containing expert trajectories.

3. (Optional) Check how the expert policy is doing:

   ```bash
   python policy_visualization.py
   ```
4. Train the policy:
    ```bash
    python diffusion_policy.py --mode train  
    # or
    python flow_matching.py --mode train
    ```
Training logs will be saved automatically and can be visualized via TensorBoard. 

This is an easy task with a lightweight model, and it can be trained within 5 minutes on an MacBook Air.

5. Evaluate The policy:
   ```bash
   # Diffusion Policy
   python diffusion_policy.py --mode inference_rgb_array  # Outputs average reward over 5 episodes
   python diffusion_policy.py --mode inference_human       # Visualizes 1 episode

   # Flow Matching Policy
   python flow_matching.py --mode inference_rgb_array      
   python flow_matching.py --mode inference_human
   ```
## References:
- Original paper: [Diffusion Policy](https://arxiv.org/abs/2303.04137v5),
                  [Flow matching](https://arxiv.org/abs/2210.02747)
- Environment: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)