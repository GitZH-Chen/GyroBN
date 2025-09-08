"""
Test: poly_blocks_to_poincare_concate <-> poincare_concate_to_poly_blocks

This script verifies that the two maps implemented in
CorPolyPoincareBallCholeskyMetric are (numerical) inverses:

  (A) blocks -> concat -> blocks
  (B) concat -> blocks -> concat

Sampling is done directly in the Poincaré ball using Stereographic.random_normal.
"""

import torch as th

from GyroBN.Geometry import Stereographic
from GyroBN.Geometry import CorPolyPoincareBallCholeskyMetric


def run_inverse_tests(
    n: int = 8,
    batch_size: int = 64,
    K_ball: float = -1.0,
    scale_blocks: float = 0.2,
    scale_concat: float = 0.2,
    tol: float = 1e-8,
    dtype=th.float64,
    device: str = "cpu",
):
    """
    Run both inverse tests:
      (A) blocks -> concat -> blocks (per i=1..n-1)
      (B) concat -> blocks -> concat
    """
    device = th.device(device)
    # geometry
    metric = CorPolyPoincareBallCholeskyMetric(n=n).to(device=device)
    pball = Stereographic(K=K_ball).to(device=device)

    # ---------------------------
    # (A) blocks -> concat -> blocks
    # ---------------------------
    # sample x_i in B^i using pball.random_normal
    xs = []
    for i in range(1, n):
        xi = pball.random_normal(batch_size, i, scale=scale_blocks).to(dtype=dtype, device=device)
        xs.append(xi)

    # forward & inverse
    z = metric.poly_blocks_to_poincare_concate(xs)                 # [..., D]
    xs_rt = metric.poincare_concate_to_poly_blocks(z, n=n)         # list of [..., i]

    # measure error per-block
    max_err_blocks = 0.0
    for i, (x1, x2) in enumerate(zip(xs, xs_rt), start=1):
        diff = (x1 - x2).abs().max().item()
        max_err_blocks = max(max_err_blocks, diff)

    # ---------------------------
    # (B) concat -> blocks -> concat
    # ---------------------------
    D = n * (n - 1) // 2
    z0 = pball.random_normal(batch_size, D, scale=scale_concat).to(dtype=dtype, device=device)  # [..., D]
    xs2 = metric.poincare_concate_to_poly_blocks(z0, n=n)
    z_rt = metric.poly_blocks_to_poincare_concate(xs2)

    max_err_concat = (z0 - z_rt).abs().max().item()

    # ---------------------------
    # Reporting
    # ---------------------------
    okA = "✅" if max_err_blocks < tol else "❌"
    okB = "✅" if max_err_concat < tol else "❌"

    print(f"\n[poly_blocks ↔ poincare_concate Inverse Test]  n={n}, B={batch_size}, K={K_ball}")
    print("--------------------------------------------------------------------------")
    print(f"{okA} blocks -> concat -> blocks   | max |Δ| = {max_err_blocks:.3e} (tol={tol:.1e})")
    print(f"{okB} concat -> blocks -> concat   | max |Δ| = {max_err_concat:.3e} (tol={tol:.1e})")
    print("--------------------------------------------------------------------------\n")


if __name__ == "__main__":
    th.manual_seed(42)
    dim=128
    run_inverse_tests(
        n=dim,
        batch_size=1000,
        K_ball=-1.0,
        scale_blocks=1/dim,
        scale_concat=1/dim,
        tol=1e-8,
        dtype=th.float64,
        device="cpu",
    )
