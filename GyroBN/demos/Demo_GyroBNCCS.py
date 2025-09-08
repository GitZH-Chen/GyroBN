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

from GyroBN.Geometry.constantcurvature import classes
from GyroBN import GyroBNCCS

# ========== Config ==========
th.manual_seed(42)
geometry = "Klein"  # "Stereographic", "Hyperboloid", "Klein", "Sphere"
K = -2.0
bs, d = 2708, 128
shape = [d]
dtype = th.float64
scale=1/d
init_1st_batch=True # Use the first batch to initialize running stats.

# ========== Generate manifold ==========
manifold = classes[geometry](K=K)



# ========== Data ==========
x = manifold.random_normal(bs, d, scale=scale).requires_grad_(True).to(dtype)
target = manifold.random_normal(bs, d, scale=scale).to(dtype)

# ========== Manifold Check (fixed) ==========
def _ok_with_msg(res):
    if isinstance(res, tuple):
        return bool(res[0]), res[1]
    return bool(res), None

ok_x, msg_x = _ok_with_msg(manifold._check_point_on_manifold(x))
ok_t, msg_t = _ok_with_msg(manifold._check_point_on_manifold(target))
if ok_x and ok_t:
    print("\033[1;36m[Check Passed]\033[0m x and target are valid manifold points.")
else:
    print("\033[1;31m[Check Failed]\033[0m",
          f"x: {msg_x if msg_x is not None else 'invalid'} ; "
          f"target: {msg_t if msg_t is not None else 'invalid'}")

# ========== Initialize GyroBN ==========
bn = GyroBNCCS(shape=shape, model=geometry, K=K, init_1st_batch=init_1st_batch).to(dtype)
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
