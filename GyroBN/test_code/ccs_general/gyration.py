import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#              GyroBN Geometry - Gyration Consistency Test
# ============================================================
#
# This script verifies the consistency of the gyration operator:
#
#     gyr[x, y](v) = ⊖(x ⊕ y) ⊕ (x ⊕ (y ⊕ v))
#
# across multiple Riemannian manifolds in GyroBN.
# Each test prints deviation statistics and ✅/❌ indicators.
# ============================================================

def run_gyration_test(manifold, batch_size=1000, dim=128, scale=None,
                      threshold=1e-8, device="cpu", dtype=torch.float64):
    """
    Test the consistency of gyration identity:
        gyr[x, y](v) = ⊖(x ⊕ y) ⊕ (x ⊕ (y ⊕ v))
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample points ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    y = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    v = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Compute both sides ----------
    lhs = manifold.gyration(x, y, v)
    rhs = manifold.gyroadd(manifold.gyroinv(manifold.gyroadd(x, y)),
                           manifold.gyroadd(x, manifold.gyroadd(y, v)))

    # ---------- Compare ----------
    diff = (lhs - rhs).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Gyration Test: gyr[x, y](v) = ⊖(x ⊕ y) ⊕ (x ⊕ (y ⊕ v))]")
    print("--------------------------------------------------")
    print(f"{result} Max deviation ‖lhs - rhs‖  = {max_diff:.3e}")
    print(f"    Mean deviation              = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_gyration_test(Sphere(K=abs_curvature))
    run_gyration_test(Stereographic(K=abs_curvature))
    run_gyration_test(Hyperboloid(K=-abs_curvature))
    run_gyration_test(Klein(K=-abs_curvature))
    run_gyration_test(Stereographic(K=-abs_curvature))
