import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#         GyroBN Geometry - Log₀ Consistency Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB ops:
#   log0(X) ≈ log(I_{n,p}, X)
#
# Each test prints max deviation with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_log0_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                            threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test: log0(X) ≈ log(I_{n,p}, X) on Gr(n,p).
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

    # ---------- Sample points ----------
    X = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    I = manifold.zero_like(X)

    okX, rX = _unpack_check(manifold._check_point_on_manifold(X))
    if not okX:
        raise ValueError(f"X is not on the manifold: {rX}")

    # ---------- Compute both log versions ----------
    U_log0 = manifold.log0(X)          # tangent at I
    U_log  = manifold.log(I, X)        # tangent at I

    is_tan0, r0 = _unpack_check(manifold._check_vector_on_tangent(I, U_log0))
    is_tan1, r1 = _unpack_check(manifold._check_vector_on_tangent(I, U_log))
    if not is_tan0:
        raise ValueError(f"log0(X) not in T_I: {r0}")
    if not is_tan1:
        raise ValueError(f"log(I, X) not in T_I: {r1}")

    # ---------- Compare (entrywise) ----------
    diff = (U_log0 - U_log).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Log₀ Consistency Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} Max ‖log₀(X) - log(I,X)‖ = {max_diff:.3e}")
    print(f"    Mean difference            = {mean_diff:.3e}")
    print(f"    % within threshold         = {pass_ratio:.2f}%")
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
    run_grassmann_log0_test(M, lead_size=(32, 8), device=device, dtype=dtype)
