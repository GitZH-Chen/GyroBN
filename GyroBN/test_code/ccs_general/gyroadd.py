import torch
from GyroBN.Geometry import Hyperboloid, Klein, Stereographic, Sphere

# ============================================================
#         GyroBN Geometry - Gyroaddition Consistency Test
# ============================================================
#
# This script verifies the gyroaddition identity:
#
#   P ⊕ Q = Exp_P(PT_{E→P}(Log_E(Q)))
#
# where E is the origin point on the manifold.
# Each test prints max difference and ✅/❌ indicators.
# ============================================================

def run_gyroadd_consistency_test(manifold, batch_size=1000, dim=128, scale=None,
                                  threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the consistency of gyroaddition via:
        P ⊕ Q ≈ Exp_P(PT_{E→P}(Log_E(Q)))
    """
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)
    manifold.K = manifold.K.to(device)

    # ---------- Sample points ----------
    P = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    Q = manifold.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    E = manifold.zero_like(P)  # origin

    # ---------- Compute each side ----------
    lhs = manifold.gyroadd(P, Q)

    log_E_Q = manifold.log0(Q)
    pt = manifold.transpfrom0(P, log_E_Q)
    rhs = manifold.exp(P, pt)

    # ---------- Compare ----------
    diff = (lhs - rhs).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[{manifold.__class__.__name__} (K={float(manifold.K):+.2f}) Gyroadd Consistency Test]")
    print("--------------------------------------------------")
    print(f"{result} P ⊕ Q ≈ Exp_P(PT_{{0→P}}(Log_0(Q)))")
    print(f"    Max deviation              = {max_diff:.3e}")
    print(f"    Mean deviation             = {mean_diff:.3e}")
    print(f"    Percentage within threshold = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    abs_curvature = 2.0

    run_gyroadd_consistency_test(Sphere(K=abs_curvature))
    run_gyroadd_consistency_test(Stereographic(K=abs_curvature))
    run_gyroadd_consistency_test(Hyperboloid(K=-abs_curvature))
    run_gyroadd_consistency_test(Klein(K=-abs_curvature))
    run_gyroadd_consistency_test(Stereographic(K=-abs_curvature))
