import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Transport-from-0 Consistency Test
# ============================================================
#
# This script verifies that:
#     transpfrom0(x, u) ≈ transp(origin, x, u)
#
# Each test prints max deviation with ✅/❌ indicators.
# ============================================================

def run_transpfrom0_test(manifold, batch_size=1000, dim=128, scale=None,
                         threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the consistency of:
        transpfrom0(x, u) ≈ transp(origin, x, u)
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample x ∈ M and u ∈ T₀M ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    u = manifold.random_tangent_origin(batch_size, dim, scale=scale).to(dtype).to(device)
    origin = manifold.zero_like(x)

    # ---------- Check validity ----------
    ok_x, reason = manifold._check_point_on_manifold(x)
    ok_u, reason_u = manifold._check_vector_on_tangent(origin, u)
    if not ok_x:
        raise ValueError(f"x is not on the manifold: {reason}")
    if not ok_u:
        raise ValueError(f"u is not in T₀(M): {reason_u}")

    # ---------- Compute two transports ----------
    t0 = manifold.transpfrom0(x, u)
    t1 = manifold.transp(origin, x, u)

    is_tan_0, _ = manifold._check_vector_on_tangent(x, t0)
    is_tan_1, _ = manifold._check_vector_on_tangent(x, t1)
    if not is_tan_0 or not is_tan_1:
        raise ValueError("Transport results not in TₓM")

    # ---------- Compare ----------
    diff = (t0 - t1).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Transport-from-0 Test]")
    print("--------------------------------------------------")
    print(f"{result} Max ‖transpfrom0 - transp‖   = {max_diff:.3e}")
    print(f"    Mean difference             = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_transpfrom0_test(Sphere(K=abs_curvature))
    run_transpfrom0_test(Stereographic(K=abs_curvature))
    run_transpfrom0_test(Hyperboloid(K=-abs_curvature))
    run_transpfrom0_test(Klein(K=-abs_curvature))
    run_transpfrom0_test(Stereographic(K=-abs_curvature))
