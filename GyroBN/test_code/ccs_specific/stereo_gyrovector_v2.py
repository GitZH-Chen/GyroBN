import torch
from GyroBN.Geometry import Stereographic

# Set reproducibility
torch.manual_seed(42)

# Initialize manifold and device
manifold = Stereographic(K=2.0)
device = torch.device("cpu")
dtype = torch.float64
manifold.K = manifold.K.to(device)

# Parameters
batch_size = 1000
dim = 128
scale = 1/dim

# Random base direction x ∈ 𝕸_K^n
x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

# Choose scalar coefficients
s = torch.tensor(0.1)
t = torch.tensor(-1.5)

# Compute each side of V2
lhs = manifold.gyroscalarprod(x, s + t)
rhs = manifold.gyroadd(manifold.gyroscalarprod(x, s), manifold.gyroscalarprod(x, t))

# Compare
max_diff = (lhs - rhs).abs().max()
print(f"[V2] Max absolute difference: {max_diff:.3e}")
