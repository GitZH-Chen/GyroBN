import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#       GyroBN Geometry - log0(exp0(u)) ≈ u Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   1) Sample u ∈ T_{I_{n,p}}(Gr)
#   2) x = Exp_0(u)
#   3) u_rec = Log_0(x)
#   4) Check u_rec ≈ u  (max-norm)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_log0exp0_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test whether log0(exp0(u)) ≈ u holds on Gr(n,p) using the ONB representation.

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

    # ---------- Sample tangent at origin (use full Δ = [0; B]) ----------
    u = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=True).to(dtype).to(device)
    origin = manifold.zero_like(u)  # I_{n,p} with same leading dims

    # ---------- Tangent validity ----------
    ok_tan, reason_t = _unpack_check(manifold._check_vector_on_tangent(origin, u))
    if not ok_tan:
        raise ValueError(f"u not in T_{origin.shape[-2:]}: {reason_t}")

    # ---------- Apply exp0 then log0 ----------
    x = manifold.exp0(u)
    ok_x, reason_x = _unpack_check(manifold._check_point_on_manifold(x))
    if not ok_x:
        raise ValueError(f"exp0(u) not on manifold: {reason_x}")

    u_rec = manifold.log0(x)
    ok_tan2, reason_t2 = _unpack_check(manifold._check_vector_on_tangent(origin, u_rec))
    if not ok_tan2:
        raise ValueError(f"log0(exp0(u)) not in T_0: {reason_t2}")

    # ---------- Compare ----------
    diff = (u - u_rec).abs().max()
    result = "✅" if diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) log0∘exp0 Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} log0(exp0(u)) ≈ u         \t Max Δ = {diff:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Example
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    n, p = 50, 10
    device = 'cpu'
    dtype = torch.float64
    M = GrassmannianGyro(n=n, p=p, exp_mode="expm").to(dtype).to(device)  # or "cayley"
    run_grassmann_log0exp0_test(M, lead_size=(32, 8), device=device, dtype=dtype)
