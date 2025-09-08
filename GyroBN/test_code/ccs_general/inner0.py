import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Inner Product at Origin Test
# ============================================================
#
# This script checks consistency between:
#   inner0(u, v) ≈ inner(origin, u, v)
# where origin = zero_like(u), i.e. base point is 0.
#
# Each test prints max deviation with ✅/❌ indicators.
# ============================================================

def run_inner0_test(manifold, batch_size=1000, dim=128, scale=None,
                    threshold=1e-8, device="cpu", dtype=torch.float64):
    """
    Test identity: inner0(u, v) ≈ inner(origin, u, v)
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample tangent vectors ----------
    u = manifold.random_tangent_origin(batch_size, dim, scale=scale).to(dtype).to(device)
    v = manifold.random_tangent_origin(batch_size, dim, scale=scale).to(dtype).to(device)
    origin = manifold.zero_like(u)

    # ---------- Compute inner products ----------
    inner_0 = manifold.inner0(u, v, keepdim=False)
    inner_ref = manifold.inner(origin, u, v, keepdim=False)

    # ---------- Compare ----------
    diff = (inner_ref - inner_0).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Inner₀ Consistency Test]")
    print("--------------------------------------------------")
    print(f"{result} Max |inner - inner0|      = {max_diff:.3e}")
    print(f"    Mean difference           = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_inner0_test(Sphere(K=abs_curvature))
    run_inner0_test(Stereographic(K=abs_curvature))
    run_inner0_test(Hyperboloid(K=-abs_curvature))
    run_inner0_test(Klein(K=-abs_curvature))
    run_inner0_test(Stereographic(K=-abs_curvature))
