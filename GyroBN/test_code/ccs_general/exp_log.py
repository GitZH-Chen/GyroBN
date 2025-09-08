import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#          GyroBN Geometry - log(exp(x, u)) ≈ u Test
# ============================================================
#
# This script verifies that applying log to exp(x, u) recovers
# the original tangent vector u at x:
#
#     log(x, exp(x, u)) ≈ u
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def run_logexp_test(manifold, batch_size=1000, dim=128, scale=None,
                    threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test whether log(exp(x, u)) ≈ u holds for a manifold.
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample point x ∈ M ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype=dtype, device=device)

    ok, reason = manifold._check_point_on_manifold(x)
    if not ok:
        raise ValueError(f"x not on manifold: {reason}")

    # ---------- Generate tangent u ∈ T_xM ----------
    raw_u = torch.randn(batch_size, manifold.dim_to_sh([dim]), dtype=dtype, device=device) / dim
    u = manifold.proju(x, raw_u)

    is_tan, reason = manifold._check_vector_on_tangent(x, u)
    if not is_tan:
        raise ValueError(f"u not in T_x(M): {reason}")

    # ---------- exp and log ----------
    y = manifold.exp(x, u)

    ok2, reason2 = manifold._check_point_on_manifold(y)
    if not ok2:
        raise ValueError(f"exp(x, u) not on manifold: {reason2}")

    u_recovered = manifold.log(x, y)

    is_tan2, reason2 = manifold._check_vector_on_tangent(x, u_recovered)
    if not is_tan2:
        raise ValueError(f"log(x, y) not in T_x(M): {reason2}")

    # ---------- Compare ----------
    diff = (u - u_recovered).abs().max()
    result = "✅" if diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) log∘exp Test]")
    print("--------------------------------------------------")
    print(f"{result} log(x, exp(x, u)) ≈ u     \t Max Δ = {diff:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_logexp_test(Sphere(K=abs_curvature))
    run_logexp_test(Stereographic(K=abs_curvature))
    run_logexp_test(Hyperboloid(K=-abs_curvature))
    run_logexp_test(Klein(K=-abs_curvature))
    run_logexp_test(Stereographic(K=-abs_curvature))
