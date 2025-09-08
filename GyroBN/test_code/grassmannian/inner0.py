import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#     GyroBN Geometry - Inner Product at Origin Test (Gr)
# ============================================================
#
# Checks on Gr(n,p) with ONB/gyro ops:
#   inner0(u, v) ≈ inner(I_{n,p}, u, v)
#
# Each test prints deviation stats with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_inner0_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                              threshold=1e-8, device="cpu", dtype=torch.float64):
    """
    Test identity: inner0(u, v) ≈ inner(I_{n,p}, u, v) on Gr(n,p).

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

    # keep samples near origin
    if scale is None:
        m = max(1, min(p, n - p))
        scale = 1.0 / m

    # ---------- Sample tangent vectors at origin (full Δ=[0;B]) ----------
    u = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=True).to(dtype).to(device)
    v = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=True).to(dtype).to(device)
    origin = manifold.zero_like(u)

    # Optional: sanity check tangency
    ok_u, ru = _unpack_check(manifold._check_vector_on_tangent(origin, u))
    ok_v, rv = _unpack_check(manifold._check_vector_on_tangent(origin, v))
    if not ok_u: raise ValueError(f"u not in T_I: {ru}")
    if not ok_v: raise ValueError(f"v not in T_I: {rv}")

    # ---------- Compute inner products ----------
    inner_0   = manifold.inner0(u, v, keepdim=False)           # Frobenius on lower block
    inner_ref = manifold.inner(origin, u, v, keepdim=False)    # canonical at I

    # ---------- Compare ----------
    diff = (inner_ref - inner_0).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Inner₀ Consistency Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} Max |inner - inner0|       = {max_diff:.3e}")
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
    run_grassmann_inner0_test(M, lead_size=(32, 8), device=device, dtype=dtype)
