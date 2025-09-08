import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#                 GyroBN Geometry - Distance Test
# ============================================================
#
# This script verifies the correctness of distance-related
# operations on a given Riemannian manifold from GyroBN:
#
#   1. Checks dist(x, y) ≈ ||log_x(y)||_x
#   2. Checks dist(x, y) ≈ ||log_0(x^{-1} ⊕ y)||_0 (gyro distance)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def run_dist_test(manifold, batch_size=1000, dim=128, scale=None,
              threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the correctness of distance-related functions on a manifold.
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sampling ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    y = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Validity checks ----------
    ok_x, reason_x = manifold._check_point_on_manifold(x)
    ok_y, reason_y = manifold._check_point_on_manifold(y)
    if not ok_x:
        raise ValueError(f"x is not on the manifold: {reason_x}")
    if not ok_y:
        raise ValueError(f"y is not on the manifold: {reason_y}")

    log_xy = manifold.log(x, y)
    is_tangent, reason = manifold._check_vector_on_tangent(x, log_xy)
    if not is_tangent:
        raise ValueError(f"log_xy not in tangent space: {reason}")

    # ---------- Dist computations ----------
    inner_log = manifold.inner(x, log_xy, log_xy)
    log_norm = torch.clamp(inner_log, min=0).sqrt()
    true_dist = manifold.dist(x, y)

    gyro_diff = manifold.gyroadd(manifold.gyroinv(x), y)
    gyro_tangent = manifold.log0(gyro_diff)
    gyro_inner = manifold.inner0(gyro_tangent, gyro_tangent)
    gyro_dist = torch.clamp(gyro_inner, min=0).sqrt()

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={manifold.K}) Distance Operator Test]")
    print("--------------------------------------------------")

    diff1 = (true_dist - log_norm).abs().max()
    diff2 = (true_dist - gyro_dist).abs().max()

    result1 = "✅" if diff1 < threshold else "❌"
    result2 = "✅" if diff2 < threshold else "❌"

    print(f"{result1} dist(x, y) ≈ ||log_x(y)||_x \t Max Δ = {diff1:.3e}")
    print(f"{result2} dist(x, y) ≈ gyro distance  \t Max Δ = {diff2:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_dist_test(Hyperboloid(K=-abs_curvature))
    run_dist_test(Klein(K=-abs_curvature))
    run_dist_test(Stereographic(K=abs_curvature))
    run_dist_test(Sphere(K=abs_curvature))
    run_dist_test(Stereographic(K=-abs_curvature))
