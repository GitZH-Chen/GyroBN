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
scale = 1 / dim

# Random points
a = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
b = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
c = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

# Scalar t
t = torch.tensor(1.3, dtype=dtype)

# Left-hand side: gyr[a, b](t ⊗ c)
lhs = manifold.gyration(a, b, manifold.gyroscalarprod(c, t))

# Right-hand side: t ⊗ gyr[a, b](c)
rhs = manifold.gyroscalarprod(manifold.gyration(a, b, c), t)

# Compare
max_diff = (lhs - rhs).abs().max()
print(f"[V4] Max absolute difference: {max_diff:.3e}")
