import torch
from GyroBN.Geometry import Klein, Stereographic  # Stereographic(K<0) is Poincaré model

# ============================================================
#           GyroBN Geometry - Klein vs Poincaré Log Test
# ============================================================
#
# This script verifies that Klein's `log` is consistent with
# the Poincaré model under isometric mapping:
#
#   1. Compute log_K(x, y) directly in Klein model.
#   2. Map x, y to Poincaré, compute log_P, then map back.
#   3. Compare the result with Klein.log(x, y).
#
# Each test prints deviation stats and ✅/❌ indicators.
# ============================================================

def run_klein_log_test(K=-4.0, batch_size=1000, dim=128, scale=None,
                       threshold=1e-8, device="cpu", dtype=torch.float64):
    # ---------- Setup ----------
    if scale is None:
        scale = 1 / dim
    device = torch.device(device)

    klein = Klein(K=K).to(dtype)
    poincare = Stereographic(K=K).to(dtype)

    # ---------- Sampling ----------
    x = klein.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)
    y = klein.random_normal(batch_size, dim, scale=scale).to(dtype).to(device)

    # ---------- Check Klein: x, y ∈ M ----------
    ok_x, reason_x = klein._check_point_on_manifold(x)
    ok_y, reason_y = klein._check_point_on_manifold(y)
    if not ok_x:
        raise ValueError(f"x not on Klein manifold: {reason_x}")
    if not ok_y:
        raise ValueError(f"y not on Klein manifold: {reason_y}")

    # ---------- Compute Klein.log ----------
    log_klein = klein.log(x, y)

    # ---------- Check log(x,y) ∈ T_x Klein ----------
    is_tangent_k, reason_k = klein._check_vector_on_tangent(x, log_klein)
    if not is_tangent_k:
        raise ValueError(f"log(x,y) not in Tₓ Klein: {reason_k}")

    # ---------- Poincaré Mapping ----------
    x_p = klein.klein_to_poincare(x)
    y_p = klein.klein_to_poincare(y)

    # ---------- Check Poincaré: x_p, y_p ∈ M ----------
    ok_xp, reason_xp = poincare._check_point_on_manifold(x_p)
    ok_yp, reason_yp = poincare._check_point_on_manifold(y_p)
    if not ok_xp:
        raise ValueError(f"x_p not on Poincaré manifold: {reason_xp}")
    if not ok_yp:
        raise ValueError(f"y_p not on Poincaré manifold: {reason_yp}")

    # ---------- Compute Poincaré.log and map back ----------
    log_p = poincare.log(x_p, y_p)
    log_expected = klein.d_poincare_to_klein(x_p, log_p)

    # ---------- Check expected log ∈ Tₓ Klein ----------
    is_tangent_expected, reason_exp = klein._check_vector_on_tangent(x, log_expected)
    if not is_tangent_expected:
        raise ValueError(f"Expected log not in Tₓ Klein: {reason_exp}")

    # ---------- Deviation ----------
    diff = (log_klein - log_expected).norm(dim=-1)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[Klein (K={K:+.2f}) log vs. Poincaré Isometry Test]")
    print("--------------------------------------------------")
    print(f"{result} Max deviation ‖log - expected‖ = {max_diff:.3e}")
    print(f"    Mean deviation                 = {mean_diff:.3e}")
    print(f"    Percentage within threshold    = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    run_klein_log_test(K=-2.0)
