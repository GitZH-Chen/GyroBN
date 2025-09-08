import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#         GyroBN Geometry - Gyroinv Consistency Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   gyroinv(X) ≈ Exp_0(−Log_0(X))
#
# Here we realize gyroinv(X) via left-translation:
#   gyroinv(X) = (−X) ⊕ I  ≡ gyroadd(X, I, is_inverse=True)
#
# Each test prints max/mean deviation and ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_gyroinv_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                               threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test the identity: gyroinv(X) ≈ Exp_0(−Log_0(X)) on Gr(n,p).

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
    okX, rX = _unpack_check(manifold._check_point_on_manifold(X))
    if not okX:
        raise ValueError(f"X not on manifold: {rX}")

    I = manifold.zero_like(X)  # I_{n,p} with same leading dims

    # ---------- Compute both sides ----------
    # lhs: gyroinv(X) ≡ (-X) ⊕ I
    lhs = manifold.gyroadd(X, I, is_inverse=True)

    # rhs: Exp_0( -Log_0(X) )
    rhs = manifold.exp0(-manifold.log0(X))

    # ---------- Compare (Frobenius per sample) ----------
    diff = (lhs - rhs)
    frob = (diff * diff).sum(dim=(-2, -1)).sqrt()
    max_diff = frob.max().item()
    mean_diff = frob.mean().item()
    pass_ratio = (frob < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Gyroinv Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} gyroinv(X) ≈ Exp_0(−Log_0(X))")
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
    run_grassmann_gyroinv_test(M, lead_size=(32, 8), device=device, dtype=dtype)
