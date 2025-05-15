# Diffusion Policy Minimal

A minimal and educational implementation of Diffusion Policy for simple control tasks like Pendulum.  
This project aims to demonstrate the core idea of applying diffusion models for action prediction.

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/ZidongChen25/Diffusion_Policy_Minimal.git
cd Diffusion_Policy_Minimal
```

### Create a Python environment (optional but recommended)

```bash
conda create -n diffusion_policy_minimal python=3.10
conda activate diffusion_policy_minimal
```

### Install required packages

```bash
pip install torch torchvision gymnasium tensorboard stable-baselines3 numpy
```

## How to Use

Diffusion Policy is an imitation learning policy, so we will need expert demonstrations, for example, human demonstration or we can train a RL policy such as PPO, SAC, DDPG. In this easy environment Pendulum, we use PPO to train an expert policy.

1. Generate expert demonstrations:

   ```bash
   python expert_policy.py
   ```

2. Create demonstrations for diffusion policy:

   ```bash
   python generate_demonstration.py
   ```

   This will create an `expert_demo.npz` file containing expert trajectories.

3. (Optional) Check how the expert policy is doing:

   ```bash
   python policy_visualization.py
   ```

4. Train the Diffusion Policy:

   ```bash
   python diffusion_policy.py --mode train
   ```

   Training logs will be saved automatically and can be visualized via TensorBoard. This is an easy task with a lightweight model, and it can be trained within 5 minutes on an M4 MacBook Air.

5. The policy will be evaluated in the Pendulum environment.
    ```bash
   python diffusion_policy.py --mode inference
   ```
---

## References

- Original Paper: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137)  <!-- Replace with actual link -->
- [Official Library](https://diffusion-policy.cs.columbia.edu/)  <!-- Replace with actual link -->
- Environment: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)