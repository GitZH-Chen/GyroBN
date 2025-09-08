"""
This is to show that the co-addition is not necessary the gyroisometry
"""

import torch
from GyroBN.Geometry import Stereographic

# Set seed
torch.manual_seed(42)

# Initialize Lorentz manifold
manifold = Stereographic(K=-2.0)
dtype = torch.float64
device = manifold.K.device


# Parameters
batch_size = 1000
dim = 128
scale = 0.1

# Generate random tangent vectors at origin
x = manifold.random_normal(batch_size,dim,scale=scale).to(device)
y = manifold.random_normal(batch_size,dim,scale=scale).to(device)
z = manifold.random_normal(batch_size,dim,scale=scale).to(device)

d1 = manifold.dist(x,y)
d2 = manifold.dist(manifold.gyrocoadd(x,z),manifold.gyrocoadd(y,z))
d3 = manifold.dist(manifold.gyrocoadd(z,x),manifold.gyrocoadd(z,y))

# left and right should be the same, w.r.t. gyrocommutative group
print(f"Max absolute difference w.r.t. right gyro coorperation: {(d1 - d2).abs().max().item():.3e}")
print(f"Max absolute difference w.r.t. left gyro coorperation: {(d1 - d3).abs().max().item():.3e}")
