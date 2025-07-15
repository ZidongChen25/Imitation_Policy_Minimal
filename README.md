
# Imitation Policy Minimal

Are you tired of constantly switching between multiple code files, original papers, and tutorials?
Frustrated by installing countless dependencies—only to run into conflicts—when all you want is to run simple demos and learn from the code?

Imitation Policy Minimal is a minimal and educational implementation of imitation learning based policies for embodied AI.
This project integrates both Diffusion Policy and Flow Matching for simple control tasks like Pendulum-v1.

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
pip install torch torchvision gymnasium tensorboard stable-baselines3 numpy
```
## How to Use

Imitation learning policy needs expert demonstrations, for example, human demonstration or we can train a RL policy such as PPO, SAC, DDPG. In this easy environment Pendulum, we use PPO to train an expert policy.

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
Training logs will be saved automatically and can be visualized via TensorBoard. This is an easy task with a lightweight model, and it can be trained within 5 minutes on an M4 MacBook Air.

5. Evaluate The policy will be evaluated in the Pendulum environment.

python diffusion_policy.py --mode inference
- Environment: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)