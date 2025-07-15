import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Define the neural ODE function (vector field)
class SpiralVectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, x):
        return self.net(x)

# Sample from a 2D standard Gaussian
n_samples = 1000
x0 = torch.randn(n_samples, 2)
t = torch.tensor([0., 1.])  # time from 0 to 1

# Define and solve the ODE
vector_field = SpiralVectorField()
with torch.no_grad():
    x_t = odeint(vector_field, x0, t)[-1]  # final state at t=1

# Convert to numpy for plotting
x0_np = x0.numpy()
x_t_np = x_t.numpy()

# Plot before and after transformation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x0_np[:, 0], x0_np[:, 1], s=5, alpha=0.5)
plt.title("Original Gaussian Samples")

plt.subplot(1, 2, 2)
plt.scatter(x_t_np[:, 0], x_t_np[:, 1], s=5, alpha=0.5, color='orange')
plt.title("Transformed Samples (Spiral CNF)")

plt.tight_layout()
plt.show()
