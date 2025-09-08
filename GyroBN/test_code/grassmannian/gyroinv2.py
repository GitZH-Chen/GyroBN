import torch
from GyroBN.Geometry import GrassmannianGyro

# ============================================================
#    GyroBN Geometry - Inversion and Identity Test (Gr)
# ============================================================
#
# Verifies on Gr(n,p) with ONB/gyro ops:
#
#       X ⊕ (⊖X) = I   and   (⊖X) ⊕ X = I
#
# with ⊖X realized as (-X) ⊕ I (left inverse via gyroadd).
# Each test prints max/mean deviation and ✅/❌ indicators.
# ============================================================

def run_grassmann_inverse_identity_test(manifold: GrassmannianGyro, lead_size=(512,), scale=None,
                                        threshold=1e-6, device="cpu", dtype=torch.float64):
    """
    Test that X ⊕ (⊖X) = I and (⊖X) ⊕ X = I on Gr(n,p).
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

    # ---------- Sample ----------
    X = manifold.random_normal(*lead_size, n, p, scale=scale).to(dtype).to(device)
    I = manifold.zero_like(X)

    # Gyro-inverse via left translation: (−X) ⊕ I
    X_inv = manifold.gyroadd(X, I, is_inverse=True)

    # ---------- Check identities ----------
    out1 = manifold.gyroadd(X, X_inv)   # should be I
    out2 = manifold.gyroadd(X_inv, X)   # should be I

    # Frobenius norm per sample over last two dims
    diff1 = ((out1 - I) ** 2).sum(dim=(-2, -1)).sqrt()
    diff2 = ((out2 - I) ** 2).sum(dim=(-2, -1)).sqrt()

    max_diff1 = diff1.max().item()
    max_diff2 = diff2.max().item()
    mean_diff1 = diff1.mean().item()
    mean_diff2 = diff2.mean().item()

    allclose1 = torch.allclose(out1, I, atol=threshold)
    allclose2 = torch.allclose(out2, I, atol=threshold)
    r1 = "✅" if allclose1 else "❌"
    r2 = "✅" if allclose2 else "❌"

    # ---------- Reporting ----------
    print(f"\n[GrassmannianGyro (n={n}, p={p}, mode={manifold.exp_mode}) Inverse Identity Test]")
    print(f"lead_size={lead_size}")
    print("--------------------------------------------------")
    print(f"{r1} X ⊕ (⊖X) ≈ I  \t Max Δ = {max_diff1:.3e} | Mean Δ = {mean_diff1:.3e}")
    print(f"{r2} (⊖X) ⊕ X ≈ I  \t Max Δ = {max_diff2:.3e} | Mean Δ = {mean_diff2:.3e}")
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
    run_grassmann_inverse_identity_test(M, lead_size=(32, 8), device=device, dtype=dtype)
