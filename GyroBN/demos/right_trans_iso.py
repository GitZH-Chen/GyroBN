"""
This is to show that the right translation is not necessary the gyroisometry
"""

import torch
from GyroBN.Geometry.constantcurvature import Stereographic

# Set seed
torch.manual_seed(42)

# Initialize manifold
manifold = Stereographic(K=-2.0)
dtype = torch.float64
device = manifold.K.device

# Parameters
batch_size = 1000
dim = 128
scale = 1/dim

# Generate random tangent vectors at origin
x = manifold.random_normal(batch_size,dim,scale=scale).to(device)
y = manifold.random_normal(batch_size,dim,scale=scale).to(device)
z = manifold.random_normal(batch_size,dim,scale=scale).to(device)

# Get Lorentz origin
d1 = manifold.dist(x,y)
d2 = manifold.dist(manifold.gyroadd(x,z),manifold.gyroadd(y,z))
d3 = manifold.dist(manifold.gyroadd(z,x),manifold.gyroadd(z,y))

print(f"Max absolute difference w.r.t. right gyroadditon: {(d1 - d2).abs().max().item():.3e}")
print(f"Max absolute difference w.r.t. left gyroadditon: {(d1 - d3).abs().max().item():.3e}")
