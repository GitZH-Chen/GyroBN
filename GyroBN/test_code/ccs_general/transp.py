import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Transport Reversibility Test
# ============================================================
#
# This script verifies the reversibility of parallel transport:
#
#   transp(y, x, transp(x, y, v)) ≈ v
#
# It also checks that:
#   - x, y ∈ M (on manifold)
#   - v ∈ T_xM, transp(x,y,v) ∈ T_yM, transp(y,x,...) ∈ T_xM
#
# Each test prints deviation statistics and ✅/❌ indicators.
# ============================================================

def run_transp_test(manifold, batch_size=1000, dim=128, scale=None,
                    threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the reversibility of parallel transport:
        transp(y, x, transp(x, y, v)) ≈ v
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample x, y ∈ M ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    y = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Validity check: x, y ∈ M ----------
    ok_x, reason_x = manifold._check_point_on_manifold(x)
    ok_y, reason_y = manifold._check_point_on_manifold(y)
    if not ok_x:
        raise ValueError(f"x not on manifold: {reason_x}")
    if not ok_y:
        raise ValueError(f"y not on manifold: {reason_y}")

    # ---------- Sample v ∈ T_xM ----------
    raw_v = torch.randn(batch_size, manifold.dim_to_sh(dim), dtype=dtype, device=device) / dim
    v = manifold.proju(x, raw_v)

    # ---------- Validity check: v ∈ T_xM ----------
    is_tan_x, reason_tan = manifold._check_vector_on_tangent(x, v)
    if not is_tan_x:
        raise ValueError(f"v not in T_xM: {reason_tan}")

    # ---------- Transport forward and back ----------
    v_to_y = manifold.transp(x, y, v)
    v_back = manifold.transp(y, x, v_to_y)

    # ---------- Validity checks: transported vectors ----------
    is_tan_y, reason_tan_y = manifold._check_vector_on_tangent(y, v_to_y)
    is_tan_x2, reason_tan_back = manifold._check_vector_on_tangent(x, v_back)
    if not is_tan_y:
        raise ValueError(f"transp(x→y) not in T_yM: {reason_tan_y}")
    if not is_tan_x2:
        raise ValueError(f"transp(y→x) not in T_xM: {reason_tan_back}")

    # ---------- Deviation ----------
    diff = (v - v_back).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Transport Reversibility Test]")
    print("--------------------------------------------------")
    print(f"{result} ‖v - transp(y,x,transp(x,y,v))‖_x")
    print(f"    Max deviation              = {max_diff:.3e}")
    print(f"    Mean deviation             = {mean_diff:.3e}")
    print(f"    Percentage within threshold= {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_transp_test(Sphere(K=abs_curvature))
    run_transp_test(Stereographic(K=abs_curvature))
    run_transp_test(Hyperboloid(K=-abs_curvature))
    run_transp_test(Klein(K=-abs_curvature))
    run_transp_test(Stereographic(K=-abs_curvature))
