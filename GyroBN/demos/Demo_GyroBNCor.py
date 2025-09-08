"""
Demo: GyroBN on Correlation Manifold via Poly Poincaré Balls

Pipeline:
  C --to_poly_blocks--> {x_i in B^i} -- per-block Poincaré GyroBN --> {x_i'}
    --from_poly_blocks--> C'
"""

import torch as th
import torch.nn as nn

from GyroBN.Geometry import CorPolyHyperbolicCholeskyMetric  # only this geometry import
from GyroBN import GyroBNCor                                    # your GyroBNCor class


# ===== Config =====
th.manual_seed(42)
dtype = th.float64
bs, n = 30, 10                    # batch size, correlation size
shape = [n, n]                   # GyroBNCor expects [..., n, n] with no channel dim

# ===== Geometry & Model =====
manifold = CorPolyHyperbolicCholeskyMetric(n=n)
bn = GyroBNCor(shape=shape).to(dtype)

# Sanity: params should be non-empty
num_params = sum(p.numel() for p in bn.parameters() if p.requires_grad)
print(f"[Model] #learnable params = {num_params}")

# ===== Data =====
x = manifold.random(bs, n, n).to(dtype).requires_grad_(True)   # [B, n, n]
target = manifold.random(bs, n, n).to(dtype)                   # [B, n, n]

# ===== Quick Check =====
cond_x = manifold._check_point_on_manifold(x)
cond_t = manifold._check_point_on_manifold(target)
def _ok(v):
    return (isinstance(v, tuple) and v[0]) or (v is True)
if _ok(cond_x) and _ok(cond_t):
    print("\033[1;36m[Check Passed]\033[0m x and target are valid correlations.")
else:
    msg_x = cond_x[1] if isinstance(cond_x, tuple) else "x invalid"
    msg_t = cond_t[1] if isinstance(cond_t, tuple) else "target invalid"
    print(f"\033[1;31m[Check Failed]\033[0m {msg_x} | {msg_t}")

# ===== Train =====
opt = th.optim.Adam(bn.parameters(), lr=1e-3)
crit = nn.MSELoss()

print("\n\033[1;32m[Training]\033[0m")
for epoch in range(5):
    bn.train()
    opt.zero_grad()
    y = bn(x)                       # [B, n, n]
    loss = crit(y, target)
    loss.backward()
    th.nn.utils.clip_grad_norm_(bn.parameters(), 1.0)
    opt.step()
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")

# ===== Eval =====
bn.eval()
with th.no_grad():
    y_eval = bn(x)                  # [B, n, n]

print("\n\033[1;34m[Eval]\033[0m")
print("Output shape:", tuple(y_eval.shape))
