import torch
from GyroBN.Geometry import Klein, Stereographic

# ============================================================
#             GyroBN Geometry - Klein vs Poincaré Exp Test
# ============================================================
#
# This script verifies that Klein's `exp` is consistent with
# the Poincaré model under isometric mapping:
#
#   1. Compute exp_K(x, u) directly in Klein model.
#   2. Map x, u to Poincaré, compute exp_P, then map back.
#   3. Compare the result with Klein.exp(x, u).
#
# Each test prints deviation stats and ✅/❌ indicators.
# ============================================================

def run_klein_exp_test(K=-4.0, batch_size=1000, dim=128, scale=None,
                       threshold=1e-8, device="cpu", dtype=torch.float64):
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)

    klein = Klein(K=K).to(dtype)
    poincare = Stereographic(K=K).to(dtype)

    # ---------- Sampling ----------
    x = klein.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    u = klein.random_tangent_origin(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Check: x ∈ Klein ----------
    ok_x, reason_x = klein._check_point_on_manifold(x)
    if not ok_x:
        raise ValueError(f"x not on Klein manifold: {reason_x}")

    # ---------- Check: u ∈ Tₓ Klein ----------
    is_tangent_k, reason_k = klein._check_vector_on_tangent(x, u)
    if not is_tangent_k:
        raise ValueError(f"u not in Tₓ Klein: {reason_k}")

    # ---------- Compute Klein.exp ----------
    exp_klein = klein.exp(x, u)

    # ---------- Check: exp(x,u) ∈ Klein ----------
    ok_exp_k, reason_exp_k = klein._check_point_on_manifold(exp_klein)
    if not ok_exp_k:
        raise ValueError(f"exp(x, u) not on Klein manifold: {reason_exp_k}")

    # ---------- Map to Poincaré ----------
    x_p = klein.klein_to_poincare(x)
    u_p = klein.d_klein_to_poincare(x, u)

    # ---------- Check: x_p ∈ Poincaré ----------
    ok_xp, reason_xp = poincare._check_point_on_manifold(x_p)
    if not ok_xp:
        raise ValueError(f"x_p not on Poincaré manifold: {reason_xp}")

    # ---------- Check: u_p ∈ Tₓ Poincaré ----------
    is_tangent_p, reason_p = poincare._check_vector_on_tangent(x_p, u_p)
    if not is_tangent_p:
        raise ValueError(f"u_p not in Tₓ Poincaré: {reason_p}")

    # ---------- Compute Poincaré.exp and map back ----------
    exp_p = poincare.exp(x_p, u_p)
    exp_expected = klein.poincare_to_klein(exp_p)

    # ---------- Check: expected ∈ Klein ----------
    ok_exp_p, reason_exp_p = klein._check_point_on_manifold(exp_expected)
    if not ok_exp_p:
        raise ValueError(f"Expected point not on Klein manifold: {reason_exp_p}")

    # ---------- Compare ----------
    diff = (exp_klein - exp_expected).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[Klein (K={K:+.2f}) exp vs. Poincaré Isometry Test]")
    print("--------------------------------------------------")
    print(f"{result} Max deviation ‖exp - expected‖ = {max_diff:.3e}")
    print(f"    Mean deviation                 = {mean_diff:.3e}")
    print(f"    Percentage within threshold    = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    run_klein_exp_test(K=-2.0)
