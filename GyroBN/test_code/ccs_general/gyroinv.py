import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Gyroinv Consistency Test
# ============================================================
#
# This script verifies the consistency of gyro-inversion:
#
#       gyroinv(x) ≈ Exp₀(−Log₀(x))
#
# across different manifolds in GyroBN. Each test prints
# max difference and ✅/❌ indicators.
# ============================================================

def run_gyroinv_test(manifold, batch_size=1000, dim=128, scale=None,
                     threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the identity: gyroinv(x) ≈ Exp₀(−Log₀(x))
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample points ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Compute both sides ----------
    lhs = manifold.gyroinv(x)
    rhs = manifold.exp0(-manifold.log0(x))

    # ---------- Compare ----------
    diff = (lhs - rhs).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Gyroinv Test: inv(x) ≈ Exp₀(−Log₀(x))]")
    print("--------------------------------------------------")
    print(f"{result} Max deviation              = {max_diff:.3e}")
    print(f"    Mean deviation             = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_gyroinv_test(Sphere(K=abs_curvature))
    run_gyroinv_test(Stereographic(K=abs_curvature))
    run_gyroinv_test(Hyperboloid(K=-abs_curvature))
    run_gyroinv_test(Klein(K=-abs_curvature))
    run_gyroinv_test(Stereographic(K=-abs_curvature))
