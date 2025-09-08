import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Gyro Scalar Product Test
# ============================================================
#
# This script verifies the identity:
#
#   gyroscalarprod(x, r) ≈ Exp₀(r · Log₀(x))
#
# across multiple manifolds in GyroBN. Each test prints
# deviation stats with ✅/❌ indicators.
# ============================================================

def run_gyroscalarprod_test(manifold, batch_size=1000, dim=128, scale=None,
                             threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the identity: gyroscalarprod(x, r) ≈ Exp₀(r · Log₀(x))
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    r = torch.randn(batch_size, 1, dtype=dtype, device=device)

    # ---------- Compute both sides ----------
    lhs = manifold.gyroscalarprod(x, r)
    rhs = manifold.exp0(r * manifold.log0(x))

    # ---------- Compare ----------
    diff = (lhs - rhs).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) GyroScalarProd Test: r ⊙ x ≈ Exp₀(r · Log₀(x))]")
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

    run_gyroscalarprod_test(Sphere(K=abs_curvature))
    run_gyroscalarprod_test(Stereographic(K=abs_curvature))
    run_gyroscalarprod_test(Hyperboloid(K=-abs_curvature))
    run_gyroscalarprod_test(Klein(K=-abs_curvature))
    run_gyroscalarprod_test(Stereographic(K=-abs_curvature))
