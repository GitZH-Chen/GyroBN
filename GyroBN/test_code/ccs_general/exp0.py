import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#            GyroBN Geometry - exp0 ≈ exp(0, u) Test
# ============================================================
#
# This script verifies the consistency between `exp0(u)` and
# `exp(origin, u)` on a given Riemannian manifold from GyroBN.
#
#   1. Samples u ∈ T_0(M)
#   2. Computes x1 = Exp_0(u), x2 = Exp(origin, u)
#   3. Checks x1 ≈ x2 in terms of max-norm difference
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def run_exp_match_test(manifold, batch_size=1000, dim=128, scale=None,
                       threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test whether exp0(u) and exp(origin, u) produce consistent results.
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample ----------
    u = manifold.random_tangent_origin(batch_size, dim, scale=scale).to(dtype=dtype, device=device)
    origin = manifold.zero_like(u)

    # ---------- Check tangent validity ----------
    is_tangent, reason = manifold._check_vector_on_tangent(origin, u)
    if not is_tangent:
        raise ValueError(f"u not in T_0(M): {reason}")

    # ---------- Compute exp and exp0 ----------
    x1 = manifold.exp0(u)
    x2 = manifold.exp(origin, u)

    # ---------- Check manifold validity ----------
    ok1, r1 = manifold._check_point_on_manifold(x1)
    ok2, r2 = manifold._check_point_on_manifold(x2)
    if not ok1:
        raise ValueError(f"exp0(u) not on manifold: {r1}")
    if not ok2:
        raise ValueError(f"exp(origin, u) not on manifold: {r2}")

    # ---------- Compare ----------
    diff = (x1 - x2).abs().max()
    result = "✅" if diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) exp0 vs exp Test]")
    print("--------------------------------------------------")
    print(f"{result} exp0(u) ≈ exp(0, u)          \t Max Δ = {diff:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_exp_match_test(Sphere(K=abs_curvature))
    run_exp_match_test(Stereographic(K=abs_curvature))
    run_exp_match_test(Hyperboloid(K=-abs_curvature))
    run_exp_match_test(Klein(K=-abs_curvature))
    run_exp_match_test(Stereographic(K=-abs_curvature))
