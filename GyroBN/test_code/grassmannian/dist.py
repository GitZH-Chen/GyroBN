import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#              GyroBN Geometry - Grassmannian Test
# ============================================================
#
# Verifies, on Gr(n,p) with ONB/gyro ops:
#   1) dist(x, y) ≈ ||log_x(y)||_x
#   2) dist(x, y) ≈ ||log_0( (-x) ⊕ y )||_0   (gyro distance)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    # _check_* may return True or (bool, reason)
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_dist_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                            threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test distance-related functions on Grassmannian with ONB gyro-ops.

    Args:
        lead_size: tuple/list/int for leading batch dims in [..., n, p].
    """
    device = torch.device(device)
    n, p = manifold.n, manifold.p

    # Normalize lead_size to a tuple
    if lead_size is None:
        lead_size = ()
    elif isinstance(lead_size, int):
        lead_size = (lead_size,)
    elif isinstance(lead_size, (list, torch.Size)):
        lead_size = tuple(lead_size)

    # ---------- Sampling ----------
    # keep samples close to origin for numerical stability
    if scale is None:
        m = max(1, min(p, n - p))
        scale = 1.0 / m

    x = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    y = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)

    # ---------- Validity checks ----------
    ok_x, reason_x = _unpack_check(manifold._check_point_on_manifold(x))
    ok_y, reason_y = _unpack_check(manifold._check_point_on_manifold(y))
    if not ok_x:
        raise ValueError(f"x is not on the manifold: {reason_x}")
    if not ok_y:
        raise ValueError(f"y is not on the manifold: {reason_y}")

    log_xy = manifold.log(x, y)
    is_tangent, reason_t = _unpack_check(manifold._check_vector_on_tangent(x, log_xy))
    if not is_tangent:
        raise ValueError(f"log_xy not in tangent space at x: {reason_t}")

    # ---------- Dist computations ----------
    inner_log = manifold.inner(x, log_xy, log_xy)         # Frobenius on the horizontal block
    log_norm  = torch.clamp(inner_log, min=0).sqrt()
    true_dist = manifold.dist(x, y)

    # Gyro distance via left translation by (-x): (-x) ⊕ y
    gyro_diff    = manifold.gyroadd(x, y, is_inverse=True)
    gyro_tangent = manifold.log0(gyro_diff)               # tangent at origin, full Δ=[0;B]
    gyro_inner   = manifold.inner0(gyro_tangent, gyro_tangent)
    gyro_dist    = torch.clamp(gyro_inner, min=0).sqrt()

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Distance Operator Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")

    diff1 = (true_dist - log_norm).abs().max()
    diff2 = (true_dist - gyro_dist).abs().max()

    result1 = "✅" if diff1 < threshold else "❌"
    result2 = "✅" if diff2 < threshold else "❌"

    print(f"{result1} dist(x, y) ≈ ||log_x(y)||_x \t Max Δ = {diff1:.3e}")
    print(f"{result2} dist(x, y) ≈ gyro distance  \t Max Δ = {diff2:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Example
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    n, p = 50, 10   # choose n > p
    device='cpu'
    dtype = torch.float64
    M = GrassmannianGyro(n=n, p=p, exp_mode="expm").to(dtype).to(device)  # cayley, expm
    run_grassmann_dist_test(M, lead_size=(30,8), device=device, dtype=dtype)
