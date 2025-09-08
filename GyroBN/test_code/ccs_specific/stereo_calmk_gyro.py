import torch
from GyroBN.Geometry import Hyperboloid, Sphere, Stereographic

# ============================================================
#        GyroBN Geometry - Isometry GyroAdd / ScalarMul Test
# ============================================================
#
# This script verifies that radius_to_stereo and stereo_to_radius
# preserve the gyrogroup structure:
#
#   - stereo(gyroadd(x, y)) = gyroadd(stereo(x), stereo(y))
#   - stereo(gyroscalarprod(x, r)) = gyroscalarprod(stereo(x), r)
#
# Each test includes:
#   ✅ Check points are on manifolds
#   ✅ Consistency of operations across isometry
# ============================================================

def run_gyro_isometry_test(K, batch_size=1000, dim=128, scale=None,
                           threshold=1e-7, device="cpu", dtype=torch.float64):
    device = torch.device(device)
    if scale is None:
        scale = 1 / dim

    stereo = Stereographic(K=K).to(dtype).to(device)
    if K < 0:
        base = Hyperboloid(K=K).to(dtype).to(device)
        label = "Hyperboloid"
    else:
        base = Sphere(K=K).to(dtype).to(device)
        label = "Sphere"

    # ---------- Sample points ----------
    x = base.random_normal(batch_size, dim, scale=scale).to(dtype)
    y = base.random_normal(batch_size, dim, scale=scale).to(dtype)
    r = torch.randn(batch_size, 1, dtype=dtype, device=device)

    # ---------- Check original manifold ----------
    for pt, name in zip([x, y], ["x", "y"]):
        ok, reason = base._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"{label}: {name} not on manifold: {reason}")

    # ---------- Isometry maps ----------
    x_st = stereo.radius_to_stereo(x)
    y_st = stereo.radius_to_stereo(y)

    # ---------- Check mapped points ----------
    for pt, name in zip([x_st, y_st], ["x_st", "y_st"]):
        ok, reason = stereo._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"Stereo: {name} not on manifold: {reason}")

    # ---------- gyroadd ----------
    g_native = base.gyroadd(x, y)
    g_stereo = stereo.gyroadd(x_st, y_st)
    g_mapped = stereo.stereo_to_radius(g_stereo)

    # Check results on manifold
    for pt, name in zip([g_native, g_mapped], ["g_native", "g_mapped"]):
        ok, reason = base._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"{label}: {name} not on manifold: {reason}")

    # ---------- gyroscalarprod ----------
    s_native = base.gyroscalarprod(x, r)
    s_stereo = stereo.gyroscalarprod(x_st, r)
    s_mapped = stereo.stereo_to_radius(s_stereo)

    for pt, name in zip([s_native, s_mapped], ["s_native", "s_mapped"]):
        ok, reason = base._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"{label}: {name} not on manifold: {reason}")

    # ---------- Compare ----------
    diff_add = (g_native - g_mapped).norm(dim=-1)
    diff_scale = (s_native - s_mapped).norm(dim=-1)

    max_add = diff_add.max().item()
    mean_add = diff_add.mean().item()
    ratio_add = (diff_add < threshold).float().mean().item() * 100

    max_scale = diff_scale.max().item()
    mean_scale = diff_scale.mean().item()
    ratio_scale = (diff_scale < threshold).float().mean().item() * 100

    pass_add = "✅" if max_add < threshold else "❌"
    pass_scale = "✅" if max_scale < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{label} (K={K:+.2f}) GyroAdd / ScalarMul Isometry Test]")
    print("--------------------------------------------------")
    print(f"{pass_add} gyroadd consistency       Max Δ = {max_add:.3e}")
    print(f"    Mean Δ = {mean_add:.3e}     Pass Rate = {ratio_add:.2f}%")
    print(f"{pass_scale} gyroscalarprod consistency Max Δ = {max_scale:.3e}")
    print(f"    Mean Δ = {mean_scale:.3e}   Pass Rate = {ratio_scale:.2f}%")
    print("--------------------------------------------------\n")

# ============================================================
#                      Run Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    run_gyro_isometry_test(K=-2.0)
    run_gyro_isometry_test(K=+2.0)
