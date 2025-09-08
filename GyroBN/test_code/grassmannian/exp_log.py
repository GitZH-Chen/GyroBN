import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#        GyroBN Geometry - log(exp(x, u)) ≈ u Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   1) Sample x ∈ Gr, u ∈ T_x Gr (via projection)
#   2) y = Exp_x(u)
#   3) Check Log_x(y) ≈ u  (max-norm)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_logexp_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                              threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test whether log(x, exp(x, u)) ≈ u holds on Gr(n,p) with ONB representation.

    Args:
        lead_size: tuple/list/int for leading batch dims in [..., n, p].
    """
    device = torch.device(device)
    n, p = manifold.n, manifold.p

    # normalize lead_size
    if lead_size is None:
        lead_size = ()
    elif isinstance(lead_size, int):
        lead_size = (lead_size,)
    elif isinstance(lead_size, (list, torch.Size)):
        lead_size = tuple(lead_size)

    # sampling scale near origin for stability
    if scale is None:
        m = max(1, min(p, n - p))
        scale = 1.0 / m

    # ---------- Sample x ∈ Gr(n,p) ----------
    x = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    ok_x, reason_x = _unpack_check(manifold._check_point_on_manifold(x))
    if not ok_x:
        raise ValueError(f"x not on manifold: {reason_x}")

    # ---------- Sample u ∈ T_x Gr via projection ----------
    # small ambient noise then project to tangent at x
    step = 1.0 / max(1, n)  # small amplitude
    raw_u = torch.randn(*lead_size, n, p, dtype=dtype, device=device) * step
    u = manifold.proju(x, raw_u)

    is_tan, reason_t = _unpack_check(manifold._check_vector_on_tangent(x, u))
    if not is_tan:
        raise ValueError(f"u not in T_x(M): {reason_t}")

    # ---------- exp and log ----------
    y = manifold.exp(x, u)
    ok_y, reason_y = _unpack_check(manifold._check_point_on_manifold(y))
    if not ok_y:
        raise ValueError(f"exp(x, u) not on manifold: {reason_y}")

    u_rec = manifold.log(x, y)
    is_tan2, reason_t2 = _unpack_check(manifold._check_vector_on_tangent(x, u_rec))
    if not is_tan2:
        raise ValueError(f"log(x, y) not in T_x(M): {reason_t2}")

    # ---------- Compare ----------
    diff = (u - u_rec).abs().max()
    result = "✅" if diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) log∘exp Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} log(x, exp(x, u)) ≈ u     \t Max Δ = {diff:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Example
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    n, p = 50, 10   # n > p
    device = 'cpu'
    dtype = torch.float64
    M = GrassmannianGyro(n=n, p=p, exp_mode="expm").to(dtype).to(device)  # or "cayley"
    run_grassmann_logexp_test(M, lead_size=(32, 8), device=device, dtype=dtype)
