import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#        GyroBN Geometry - Gyroaddition Consistency Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   P ⊕ Q ≈ Exp_P( PT_{I→P}( Log_I(Q) ) )
#
# Each test prints max/mean deviation and ✅/❌ indicators.
# ============================================================

def _unpack_check(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return bool(result), None

def run_grassmann_gyroadd_consistency_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                           threshold=1e-5, device="cpu", dtype=torch.float64):
    """
    Test P ⊕ Q ≈ Exp_P( PT_{I→P}( Log_I(Q) ) ) on Gr(n,p).

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
    P = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    Q = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    E = manifold.zero_like(P)  # I_{n,p} with same leading dims

    okP, rP = _unpack_check(manifold._check_point_on_manifold(P))
    okQ, rQ = _unpack_check(manifold._check_point_on_manifold(Q))
    if not okP: raise ValueError(f"P not on manifold: {rP}")
    if not okQ: raise ValueError(f"Q not on manifold: {rQ}")

    # ---------- Left-hand side ----------
    lhs = manifold.gyroadd(P, Q)  # P ⊕ Q

    # ---------- Right-hand side ----------
    # Log at identity: get lower block B; build zero top block for identity-case PT
    B = manifold.log0(Q, is_lower_part=True)  # (..., n-p, p)
    zero = torch.zeros(*lead_size, p, p, dtype=dtype, device=device)
    # Parallel transport Δ=[0;B] from I to P (returns full Δ_P ∈ T_P)
    delta_P = manifold.transpfrom0(P, B, zero)  # (..., n, p)
    rhs = manifold.exp(P, delta_P)

    # ---------- Compare ----------
    diff = (lhs - rhs)
    frob = (diff * diff).sum(dim=(-2, -1)).sqrt()  # Frobenius per sample
    max_diff = frob.max().item()
    mean_diff = frob.mean().item()
    pass_ratio = (frob < threshold).float().mean().item() * 100.0
    ok = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Gyroadd Consistency Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{ok}  P ⊕ Q ≈ Exp_P(PT_{{I→P}}(Log_I(Q)))")
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
    run_grassmann_gyroadd_consistency_test(M, lead_size=(32, 8), device=device, dtype=dtype)
