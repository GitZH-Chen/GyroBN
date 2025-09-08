import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#        GyroBN Geometry - exp0 ≈ exp(I_{n,p}, u) Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   1) Sample u ∈ T_{I_{n,p}}(Gr), x1 = Exp_0(u), x2 = Exp(I, u)
#   2) Check x1 ≈ x2   (max-norm)
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_exp_match_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                 threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test consistency between exp0(u) and exp(I_{n,p}, u) on the Grassmannian.

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

    # ----- sample tangent u at origin (full Δ = [0; B]) -----
    u = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=True).to(dtype).to(device)
    origin = manifold.zero_like(u)  # I_{n,p} with same batch shape

    # tangent validity at origin
    ok_tan, reason_t = _unpack_check(manifold._check_vector_on_tangent(origin, u))
    if not ok_tan:
        raise ValueError(f"u not in T_{origin.shape[-2:]}: {reason_t}")

    # maps
    x1 = manifold.exp0(u)
    x2 = manifold.exp(origin, u)

    # manifold validity
    ok1, r1 = _unpack_check(manifold._check_point_on_manifold(x1))
    ok2, r2 = _unpack_check(manifold._check_point_on_manifold(x2))
    if not ok1:
        raise ValueError(f"exp0(u) not on manifold: {r1}")
    if not ok2:
        raise ValueError(f"exp(I, u) not on manifold: {r2}")

    # compare
    diff = (x1 - x2).abs().max()
    result = "✅" if diff < threshold else "❌"

    # report
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) exp0 vs exp(I, u) Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} exp0(u) ≈ exp(I, u)          \t Max Δ = {diff:.3e}")
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
    run_grassmann_exp_match_test(M, lead_size=(32, 8), device=device, dtype=dtype)
