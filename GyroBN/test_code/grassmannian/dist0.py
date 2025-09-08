import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#            GyroBN Geometry - exp0 vs. dist0 Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   1) Sample u ∈ T_{I_{n,p}}(Gr), x = Exp_0(u)
#   2) Check dist(I, x) ≈ dist0(x) ≈ ||u||_0
#
# Each test prints max difference with ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_exp0_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                            threshold=1e-6, device="cpu", dtype=torch.float64, keepdim=True):
    """
    Test the correctness of exp0 and dist0 on Gr(n,p) with ONB representation.

    Args:
        lead_size: tuple/list/int for leading batch dims in [..., n, p].
    """
    device = torch.device(device)
    n, p = manifold.n, manifold.p

    # Normalize lead_size
    if lead_size is None:
        lead_size = ()
    elif isinstance(lead_size, int):
        lead_size = (lead_size,)
    elif isinstance(lead_size, (list, torch.Size)):
        lead_size = tuple(lead_size)

    # Sampling scale near origin for stability
    if scale is None:
        m = max(1, min(p, n - p))
        scale = 1.0 / m

    # ---------- Sample tangent vectors at origin (full Δ=[0;B]) ----------
    u = manifold.random_tangent_origin(*lead_size, n, p, scale=scale, full_matrix=True).to(dtype).to(device)
    origin = manifold.zero_like(u)

    # ---------- Check tangent validity ----------
    ok_tan, reason_t = _unpack_check(manifold._check_vector_on_tangent(origin, u))
    if not ok_tan:
        raise ValueError(f"u not in T_{origin.shape[-2:]}: {reason_t}")

    # ---------- Exponential map ----------
    x = manifold.exp0(u)

    # ---------- Check manifold validity ----------
    ok_x, reason_x = _unpack_check(manifold._check_point_on_manifold(x))
    if not ok_x:
        raise ValueError(f"exp0(u) not on manifold: {reason_x}")

    # ---------- Compare distances ----------
    d1 = manifold.dist(origin, x, keepdim=keepdim)   # geodesic dist to I_{n,p}
    d2 = manifold.dist0(x, keepdim=keepdim)          # dist0(x) via ONB upper block

    # ---------- Compute norm ||u||_0 ----------
    u_inner = manifold.inner0(u, u, keepdim=keepdim) # Frobenius on lower (n-p)×p block
    u_norm  = torch.clamp(u_inner, min=0).sqrt()

    # ---------- Compare ----------
    diff_I_vs_d0 = (d1 - d2).abs().max()
    diff_u_vs_d0 = (u_norm - d2).abs().max()

    r1 = "✅" if diff_I_vs_d0 < threshold else "❌"
    r2 = "✅" if diff_u_vs_d0 < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) exp0 Test]")
    print("--------------------------------------------------")
    print(f"{r1} dist(I, x) ≈ dist0(x)           \t Max Δ = {diff_I_vs_d0:.3e}")
    print(f"{r2} ||u||_0 ≈ dist0(exp0(u))        \t Max Δ = {diff_u_vs_d0:.3e}")
    print("--------------------------------------------------\n")


# ============================================================
#                      Run Example
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    n, p = 50, 10    # choose n > p
    device = 'cpu'
    dtype = torch.float64
    M = GrassmannianGyro(n=n, p=p, exp_mode="expm").to(dtype).to(device)  # or "cayley"
    run_grassmann_exp0_test(M, lead_size=(32, 8), device=device, dtype=dtype)
