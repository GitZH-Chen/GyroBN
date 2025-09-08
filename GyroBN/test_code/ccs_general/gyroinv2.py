import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Inversion and Identity Test
# ============================================================
#
# This script verifies the identity:
#
#       x ⊕ (⊖x) = 0     and     (⊖x) ⊕ x = 0
#
# for the gyro inverse and addition operators on each manifold.
# Each test prints max difference and ✅/❌ indicators.
# ============================================================

def run_gyroinv_test(manifold, batch_size=1000, dim=128, scale=None,
                     threshold=1e-7, device="cpu", dtype=torch.float64):
    """
    Test that x ⊕ (⊖x) = 0 and (⊖x) ⊕ x = 0 for the manifold.
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample ----------
    x = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    x_inv = manifold.gyroinv(x)
    zero = manifold.zero_like(x)

    # ---------- Compute x ⊕ (⊖x) and (⊖x) ⊕ x ----------
    out1 = manifold.gyroadd(x, x_inv)
    out2 = manifold.gyroadd(x_inv, x)

    diff1 = (out1 - zero).norm(dim=-1)
    diff2 = (out2 - zero).norm(dim=-1)

    max_diff1 = diff1.max().item()
    max_diff2 = diff2.max().item()
    allclose1 = torch.allclose(out1, zero, atol=threshold)
    allclose2 = torch.allclose(out2, zero, atol=threshold)

    r1 = "✅" if allclose1 else "❌"
    r2 = "✅" if allclose2 else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Inverse Identity Test]")
    print("--------------------------------------------------")
    print(f"{r1} x ⊕ (⊖x) ≈ 0  \t Max Δ = {max_diff1:.3e}")
    print(f"{r2} (⊖x) ⊕ x ≈ 0  \t Max Δ = {max_diff2:.3e}")
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
