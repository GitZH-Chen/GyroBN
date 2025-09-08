import torch
from GyroBN.Geometry import Hyperboloid

def run_gyro_closed_vs_isometry(K, batch_size=1000, dim=128, scale=None,
                                threshold=1e-7, device="cpu", dtype=torch.float64):
    """
    测试 Hyperboloid.gyroadd_radius_closed 与 Hyperboloid.gyroadd 是否一致
    """
    device = torch.device(device)
    if scale is None:
        scale = 1 / dim

    base = Hyperboloid(K=K).to(dtype).to(device)
    label = "Hyperboloid"

    # ---------- Sample points ----------
    x = base.random_normal(batch_size, dim, scale=scale).to(dtype)
    y = base.random_normal(batch_size, dim, scale=scale).to(dtype)

    # ---------- Check original manifold ----------
    for pt, name in zip([x, y], ["x", "y"]):
        ok, reason = base._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"{label}: {name} not on manifold: {reason}")

    # ---------- 两种方法计算 ----------
    g_iso = base.gyroadd(x, y)                 # 现有 isometry 实现
    g_closed = base.gyroadd_radius_closed(x, y)  # 闭式公式实现

    # ---------- Check results on manifold ----------
    for pt, name in zip([g_iso, g_closed], ["g_iso", "g_closed"]):
        ok, reason = base._check_point_on_manifold(pt)
        if not ok:
            raise ValueError(f"{label}: {name} not on manifold: {reason}")

    # ---------- Compare ----------
    diff_add = (g_iso - g_closed).norm(dim=-1)

    max_add = diff_add.max().item()
    mean_add = diff_add.mean().item()
    ratio_add = (diff_add < threshold).float().mean().item() * 100
    pass_add = "✅" if max_add < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{label} (K={K:+.2f}) GyroAdd Closed-form vs Isometry Test]")
    print("--------------------------------------------------")
    print(f"{pass_add} gyroadd consistency       Max Δ = {max_add:.3e}")
    print(f"    Mean Δ = {mean_add:.3e}     Pass Rate = {ratio_add:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    run_gyro_closed_vs_isometry(K=-2.0)
