import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Log₀ Consistency Test
# ============================================================
#
# This script verifies that log0(x) ≈ log(origin, x)
# where origin = zero_like(x)
#
# Each test prints max deviation with ✅/❌ indicators.
# ============================================================

def run_log0_test(manifold, batch_size=1000, dim=128, scale=None,
                  threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the consistency of:
        log0(x) ≈ log(origin, x)
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample points ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    origin = manifold.zero_like(x)

    # ---------- Validity checks ----------
    ok, reason = manifold._check_point_on_manifold(x)
    if not ok:
        raise ValueError(f"x is not on the manifold: {reason}")

    # ---------- Compute both log versions ----------
    u_log0 = manifold.log0(x)
    u_log = manifold.log(origin, x)

    is_tan_0, reason0 = manifold._check_vector_on_tangent(origin, u_log0)
    is_tan, reason = manifold._check_vector_on_tangent(origin, u_log)

    if not is_tan_0:
        raise ValueError(f"log0(x) not in T_0(M): {reason0}")
    if not is_tan:
        raise ValueError(f"log(origin, x) not in T_0(M): {reason}")

    # ---------- Compare ----------
    diff = (u_log0 - u_log).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Log₀ Consistency Test]")
    print("--------------------------------------------------")
    print(f"{result} Max ‖log₀(x) - log(0,x)‖     = {max_diff:.3e}")
    print(f"    Mean difference             = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_log0_test(Sphere(K=abs_curvature))
    run_log0_test(Stereographic(K=abs_curvature))
    run_log0_test(Hyperboloid(K=-abs_curvature))
    run_log0_test(Klein(K=-abs_curvature))
    run_log0_test(Stereographic(K=-abs_curvature))
