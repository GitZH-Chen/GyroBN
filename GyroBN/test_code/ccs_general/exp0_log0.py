import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#          GyroBN Geometry - log0(exp0(u)) ≈ u Test
# ============================================================
#
# This script verifies that applying log0 to exp0(u) returns
# the original tangent vector u:
#
#     log0(exp0(u)) ≈ u
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def run_log0exp0_test(manifold, batch_size=1000, dim=128, scale=None,
                      threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test whether log0(exp0(u)) ≈ u holds for a manifold.
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

    # ---------- Apply exp0 then log0 ----------
    x = manifold.exp0(u)

    ok, reason = manifold._check_point_on_manifold(x)
    if not ok:
        raise ValueError(f"exp0(u) not on manifold: {reason}")

    u_recovered = manifold.log0(x)

    is_tangent2, reason2 = manifold._check_vector_on_tangent(origin, u_recovered)
    if not is_tangent2:
        raise ValueError(f"log0(exp0(u)) not in T_0(M): {reason2}")

    # ---------- Compare ----------
    diff = (u - u_recovered).abs().max()
    result = "✅" if diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) log0∘exp0 Test]")
    print("--------------------------------------------------")
    print(f"{result} log0(exp0(u)) ≈ u         \t Max Δ = {diff:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_log0exp0_test(Sphere(K=abs_curvature))
    run_log0exp0_test(Stereographic(K=abs_curvature))
    run_log0exp0_test(Hyperboloid(K=-abs_curvature))
    run_log0exp0_test(Klein(K=-abs_curvature))
    run_log0exp0_test(Stereographic(K=-abs_curvature))
