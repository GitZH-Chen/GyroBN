import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic

# Set reproducibility
torch.manual_seed(42)

# Initialize manifold and device
manifold = Stereographic(K=2.0)
device = torch.device("cpu")  # 明确指定设备
dtype = torch.float64

# Move manifold parameters to correct device
manifold.K = manifold.K.to(device)

# Parameters
batch_size = 100
dim = 30
scale = 1 / dim

dim_out=128

# Sample two points x, y ∈ 𝕳_K^n
x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
weight = torch.randn(dim_out,dim).to(dtype).to(device)

res1 = manifold.gyro_matvec(weight, x)
res2 = manifold.exp0(torch.tensordot(manifold.log0(x), weight, ([-1], [1])))

# Compare
max_diff = (res1 - res2).abs().max()
print(f"[✓] Max absolute difference between dist(x, y) and ||log_x y||_x: {max_diff:.3e}")
