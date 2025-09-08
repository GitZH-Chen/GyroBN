import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#         GyroBN Geometry - Gyro Scalar Product Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#   gyroscalarprod(r, X) ≈ Exp₀(r · Log₀(X))
#
# Each test prints deviation stats with ✅/❌ indicators.
# ============================================================

def run_grassmann_gyroscalarprod_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                      threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test identity: gyroscalarprod(r, X) ≈ Exp₀(r · Log₀(X)) on Gr(n,p).

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

    # keep samples close to origin for numerical stability
    if scale is None:
        m = max(1, min(p, n - p))
        scale = 1.0 / m

    # ---------- Sample ----------
    X = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    # r has same leading dims, broadcasts over (n, p)
    r = torch.randn(*lead_size, 1, 1, dtype=dtype, device=device)

    # ---------- Compute both sides ----------
    lhs = manifold.gyroscalarprod(r, X)              # r ⊙ X
    rhs = manifold.exp0(r * manifold.log0(X))        # Exp₀(r · Log₀(X))

    # ---------- Compare (Frobenius per sample) ----------
    diff = ((lhs - rhs) ** 2).sum(dim=(-2, -1)).sqrt()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pass_ratio = (diff < threshold).float().mean().item() * 100.0
    result = "✅" if max_diff < threshold else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) GyroScalarProd Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{result} r ⊙ X ≈ Exp₀(r · Log₀(X))")
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
    run_grassmann_gyroscalarprod_test(M, lead_size=(32, 8), device=device, dtype=dtype)
