import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#             GyroBN Geometry - exp0 vs. dist0 Test
# ============================================================
#
# This script verifies the correctness of `exp0` and `dist0`
# on a given Riemannian manifold from GyroBN.
#
#   1. Samples u ∈ T_0(M), then maps to x = Exp_0(u)
#   2. Checks that dist(0, x) ≈ ||u||_0 = dist0(x)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def run_exp0_test(manifold, batch_size=1000, dim=128, scale=None,
                  threshold=1e-6, device="cpu", dtype=torch.float64, keepdim=True):
    """
    Test the correctness of exp0 and dist0 on a manifold.
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample tangent vectors at origin ----------
    u = manifold.random_tangent_origin(batch_size, dim, scale=scale).to(dtype=dtype, device=device)
    origin = manifold.zero_like(u)

    # ---------- Check tangent validity ----------
    is_tangent, reason = manifold._check_vector_on_tangent(origin, u)
    if not is_tangent:
        raise ValueError(f"u not in T_0(M): {reason}")

    # ---------- Exponential map ----------
    x = manifold.exp0(u)

    # ---------- Check manifold validity ----------
    ok, reason = manifold._check_point_on_manifold(x)
    if not ok:
        raise ValueError(f"exp0(u) not on manifold: {reason}")

    # ---------- Compare distances ----------
    d1 = manifold.dist(origin, x, keepdim=keepdim)
    d2 = manifold.dist0(x, keepdim=keepdim)

    # ---------- Compute norm ||u||_0 ----------
    u_inner = manifold.inner0(u, u, keepdim=keepdim)
    u_norm = torch.clamp(u_inner, min=0).sqrt()

    # ---------- Compare ----------
    diff1 = (d1 - d2).abs().max()
    diff2_all = (u_norm - d2).abs()
    diff2 = diff2_all.max()
    r1 = "✅" if diff1 < threshold else "❌"
    r2 = "✅" if diff2 < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) exp0 Test]")
    print("--------------------------------------------------")
    print(f"{r1} dist(0, x) ≈ dist0(x)           \t Max Δ = {diff1:.3e}")
    print(f"{r2} ||u||_0 ≈ dist0(exp0(u))        \t Max Δ = {diff2:.3e}")

    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_exp0_test(Sphere(K=abs_curvature))
    run_exp0_test(Stereographic(K=abs_curvature))
    run_exp0_test(Hyperboloid(K=-abs_curvature))
    run_exp0_test(Klein(K=-abs_curvature))
    run_exp0_test(Stereographic(K=-abs_curvature))

