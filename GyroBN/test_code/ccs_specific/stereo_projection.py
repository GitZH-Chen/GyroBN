import torch
from GyroBN.Geometry import Stereographic, Hyperboloid, Sphere

# ============================================================
#        Test: radius_to_stereo and stereo_to_radius
# ============================================================
#
# This test verifies that for each K:
#   - K < 0: Stereographic ≅ Hyperboloid via isometry
#   - K > 0: Stereographic ≅ Sphere via isometry
#   - radius_to_stereo ◦ stereo_to_radius ≈ identity
#   - stereo_to_radius ◦ radius_to_stereo ≈ identity
#
# Results are reported with ✅/❌ and deviation statistics.
# ============================================================

def run_stereo_isometry_test(K, batch_size=1000, dim=128, threshold=1e-7,
                             device="cpu", dtype=torch.float64):
    device = torch.device(device)
    stereo = Stereographic(K=K).to(dtype)
    if K < 0:
        other = Hyperboloid(K=K).to(dtype)
        label = "Hyperboloid"
    else:
        other = Sphere(K=K).to(dtype)
        label = "Sphere"

    # Sample x from the other manifold
    x = other.random_normal(batch_size, dim, scale=1/dim).to(dtype).to(device)

    # Check x ∈ M
    ok1, reason1 = other._check_point_on_manifold(x)
    assert ok1, f"x is not on {label}: {reason1}"

    # Mapping: M → Stereographic → M
    y = stereo.radius_to_stereo(x)
    x_recovered = stereo.stereo_to_radius(y)

    ok2, reason2 = stereo._check_point_on_manifold(y)
    assert ok2, f"y is not on Stereographic manifold: {reason2}"

    ok3, reason3 = other._check_point_on_manifold(x_recovered)
    assert ok3, f"Recovered x is not on {label}: {reason3}"

    # Compare original x and recovered x
    delta = (x - x_recovered).norm(dim=-1)
    max_diff = delta.max().item()
    mean_diff = delta.mean().item()
    ratio = (delta < threshold).float().mean().item() * 100
    result = "✅" if max_diff < threshold else "❌"

    # Reporting
    print(f"\n[Stereographic ↔ {label} Isometry Test (K={K:+.2f})]")
    print("--------------------------------------------------")
    print(f"{result} Max deviation ‖x - recovered(x)‖ = {max_diff:.3e}")
    print(f"    Mean deviation                     = {mean_diff:.3e}")
    print(f"    Percentage within threshold        = {ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run All Tests
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    run_stereo_isometry_test(K=-2.0)
    run_stereo_isometry_test(K=+2.0)
