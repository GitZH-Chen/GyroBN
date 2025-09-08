"""
Demo: GyroBN on Hyperbolic Manifolds

This script demonstrates the use of the GyroBN layer on different models of hyperbolic space:
    Poincaré ball, Hyperboloid, and Klein model.

The script:
1. Initializes a manifold (e.g., Hyperboloid with curvature K = -2).
2. Generates synthetic tangent vectors and maps them onto the manifold.
3. Applies GyroBN to normalize these points.
4. Optimizes the parameters to minimize the MSE loss between output and target.
5. Prints training loss and evaluation statistics.

Usage:
- Adjust model_name to test different models: "Poincare", "Hyperboloid", or "Klein".

Author: Ziheng Chen @ Unitn (2025)
"""

import torch as th
import torch.nn as nn

from GyroBN.Geometry import GrassmannianGyro
from GyroBN import GyroBNGr

# ========== Config ==========
th.manual_seed(42)
geometry = "Grassmannian"
bs, n, p = 30, 50, 10
shape = [n, p]
batchdim=[0]

# bs, c, n, p = 30, 8, 50, 10
# shape = [c, n, p]
# batchdim=[0]
dtype = th.float64
# scale=1/d

# ========== Generate manifold ==========
manifold = GrassmannianGyro(n=n,p=p)

# ========== Data ==========
x = manifold.random_normal(bs, n, p).requires_grad_(True).to(dtype)
target = manifold.random_normal(bs, n, p).to(dtype)

# ========== Manifold Check ==========
try:
    manifold._check_point_on_manifold(x)
    manifold._check_point_on_manifold(target)
    print("\033[1;36m[Check Passed]\033[0m x and target are valid manifold points.")
except AssertionError as e:
    print("\033[1;31m[Check Failed]\033[0m", str(e))

# ========== Initialize GyroBN ==========
bn = GyroBNGr(shape=shape, batchdim=batchdim, max_iter=100).to(dtype)
print(f"\n\033[1;34m[Model Info]\033[0m\n{bn}")
print("\n\033[1;34m[Parameters]\033[0m")
for name, param in bn.named_parameters():
    print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ========== Optimization ==========
optimizer = th.optim.Adam(bn.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("\n\033[1;32m[Training Start]\033[0m")
for epoch in range(5):
    bn.train()
    optimizer.zero_grad()

    out = bn(x)
    loss = criterion(out, target)

    loss.backward()
    grad_norm = th.nn.utils.clip_grad_norm_(bn.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"Epoch {epoch + 1:2d}: Loss = {loss.item():.6f}, GradNorm = {grad_norm:.6f}")

# ========== Evaluation ==========
bn.eval()
with th.no_grad():
    out_eval = bn(x)

print("\n\033[1;34m[Evaluation Result]\033[0m")
print(f"Output shape = {out_eval.shape}")
