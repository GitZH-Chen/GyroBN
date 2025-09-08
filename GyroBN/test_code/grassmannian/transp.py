import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#      GyroBN Geometry - Transport Reversibility Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB ops:
#   transp(Y, X, transp(X, Y, V)) ≈ V
# and checks tangency at X and Y.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_transp_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                              threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test reversibility of the vector transport on Gr(n,p):
        transp(Y, X, transp(X, Y, V)) ≈ V
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

    # ---------- Sample X, Y ∈ Gr ----------
    X = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    Y = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)

    okX, rX = _unpack_check(manifold._check_point_on_manifold(X))
    okY, rY = _unpack_check(manifold._check_point_on_manifold(Y))
    if not okX: raise ValueError(f"X not on manifold: {rX}")
    if not okY: raise ValueError(f"Y not on manifold: {rY}")

    # ---------- Sample V ∈ T_X Gr via projection ----------
    step = 1.0 / max(1, n)
    raw_V = torch.randn(*lead_size, n, p, dtype=dtype, device=device) * step
    V = manifold.proju(X, raw_V)

    is_tan_X, rTX = _unpack_check(manifold._check_vector_on_tangent(X, V))
    if not is_tan_X: raise ValueError(f"V not in T_XM: {rTX}")

    # ---------- Transport forward and back ----------
    V_to_Y = manifold.transp(X, Y, V)
    V_back = manifold.transp(Y, X, V_to_Y)

    # ---------- Tangency checks ----------
    is_tan_Y, rTY = _unpack_check(manifold._check_vector_on_tangent(Y, V_to_Y))
    is_tan_X2, rTX2 = _unpack_check(manifold._check_vector_on_tangent(X, V_back))
    if not is_tan_Y:  raise ValueError(f"transp(X→Y,V) not in T_YM: {rTY}")
    if not is_tan_X2: raise ValueError(f"transp(Y→X,...) not in T_XM: {rTX2}")

    # ---------- Deviation (Frobenius per sample over last two dims) ----------
    diff = ((V - V_back) ** 2).sum(dim=(-2, -1)).sqrt()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Transport Reversibility Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} ‖V - transp(Y,X,transp(X,Y,V))‖_F")
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
    run_grassmann_transp_test(M, lead_size=(32, 8), device=device, dtype=dtype)
