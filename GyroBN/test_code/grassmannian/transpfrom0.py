import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#      GyroBN Geometry - Transport-from-0 Consistency (Gr)
# ============================================================
#
# Verifies on Gr(n,p):
#   transpfrom0(X, B) ≈ transp(I, X, Δ=[0;B])
#
# Each test prints deviation stats with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_transpfrom0_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                   threshold=1e-5, device="cpu", dtype=torch.float64):
    """
    Test consistency of:
        transpfrom0(X, B) ≈ transp(I, X, Δ=[0;B])
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

    # ---------- Sample X ∈ Gr(n,p) ----------
    X = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    okX, rX = _unpack_check(manifold._check_point_on_manifold(X))
    if not okX:
        raise ValueError(f"X is not on the manifold: {rX}")

    # ---------- Sample u ∈ T_I Gr as lower block B and full Δ=[0;B] ----------
    B = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=False).to(dtype).to(device)  # (..., n-p, p)
    zero = torch.zeros(*lead_size, p, p, dtype=dtype, device=device)
    Delta = torch.cat([zero, B], dim=-2)  # Δ=[0;B], shape (..., n, p)

    # Tangency at origin
    I = manifold.zero_like(X)
    ok_tan0, r0 = _unpack_check(manifold._check_vector_on_tangent(I, Delta))
    if not ok_tan0:
        raise ValueError(f"Δ not in T_I: {r0}")

    # ---------- Compute both transports to T_X ----------
    t0 = manifold.transpfrom0(X, B, zero)   # expects (X, B, zero)
    t1 = manifold.transp(I, X, Delta)       # LC-PT along geodesic if you implemented it

    # Tangency at X
    ok_t0, r_t0 = _unpack_check(manifold._check_vector_on_tangent(X, t0))
    ok_t1, r_t1 = _unpack_check(manifold._check_vector_on_tangent(X, t1))
    if not ok_t0: raise ValueError(f"transpfrom0 result not in T_X: {r_t0}")
    if not ok_t1: raise ValueError(f"transp result not in T_X: {r_t1}")

    # ---------- Compare (Frobenius per sample over last two dims) ----------
    diff = ((t0 - t1) ** 2).sum(dim=(-2, -1)).sqrt()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Transport-from-0 Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} ‖transpfrom0(X,B) - transp(I,X,Δ)‖_F")
    print(f"    Max deviation               = {max_diff:.3e}")
    print(f"    Mean deviation              = {mean_diff:.3e}")
    print(f"    % within threshold          = {pass_ratio:.2f}%")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Example
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    n, p = 50, 10
    device = "cpu"
    dtype = torch.float64
    M = GrassmannianGyro(n=n, p=p, exp_mode="expm").to(dtype).to(device)  # or "cayley"
    run_grassmann_transpfrom0_test(M, lead_size=(32, 8), device=device, dtype=dtype)
