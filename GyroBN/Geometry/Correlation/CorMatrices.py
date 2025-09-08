import torch as th
from typing import List, Tuple, Optional, Union
from scipy.special import beta

from ..base import Manifold
from .ppbcm_functionals import hemisphere_to_poincare, poincare_to_hemisphere
from GyroBN.Geometry import Stereographic


class Correlation(Manifold):
    """Correlation data on [..., n, n]."""
    def __init__(self, n: int, jitter: float = 1e-6):
        super().__init__()
        assert n >= 2, "Correlation size n must be >= 2."
        self.n = n
        self.dim = n * (n - 1) // 2
        self.jitter = float(jitter)
        self.register_buffer("I", th.eye(n))

    def _check_point_on_manifold(self, C: th.Tensor, atol: float = 1e-6, rtol: float = 1e-6,
            tol_spd: float = 1e-12) -> Union[Tuple[bool, Optional[str]], bool]:
        # square
        if C.shape[-1] != C.shape[-2]:
            return False, "Matrix is not square."

        # symmetric
        if not th.allclose(C, C.transpose(-2, -1), atol=atol, rtol=rtol):
            return False, "Matrix is not symmetric."

        # diag ≈ 1
        diag = th.diagonal(C, dim1=-2, dim2=-1)
        if not th.allclose(diag, th.ones_like(diag), atol=atol, rtol=rtol):
            return False, "Diagonal not equal to 1."

        # PSD (tolerant)
        eigs = th.linalg.eigvalsh(C)
        if not th.all(eigs >= -tol_spd):
            return False, f"Not PSD (min eigenvalue={eigs.min().item():.2e})."

        # PD (Cholesky)
        try:
            L = th.linalg.cholesky(C)
        except RuntimeError:
            return False, "Cholesky failed (not PD)."

        # chol diag > 0
        if not th.all(th.diagonal(L, dim1=-2, dim2=-1) > 0):
            return False, "Cholesky diagonal has non-positive entries."

        # optional: L row norms ≈ 1 (implied by diag(C)=1, but we check tolerance)
        row_norms = (L ** 2).sum(dim=-1)
        if not th.allclose(row_norms, th.ones_like(row_norms), atol=atol, rtol=rtol):
            return False, "Cholesky row norms deviate from 1."

        return True

    def symmetrize(self, X: th.Tensor) -> th.Tensor:
        return 0.5 * (X + X.transpose(-1, -2))

    def random(self, *shape, eps: float = 1e-6) -> th.Tensor:
        """Random correlation via SPD → correlation."""
        assert len(shape) >= 2 and shape[-2] == shape[-1], "Shape must be [..., n, n]"
        n = shape[-1]
        A = th.randn(shape) * 2 - 1
        cov = A @ A.transpose(-2, -1) + eps * th.eye(n, device=A.device, dtype=A.dtype)
        return self.covariance_to_correlation(cov)

    def covariance_to_correlation(self, cov: th.Tensor) -> th.Tensor:
        d = th.diagonal(cov, dim1=-2, dim2=-1)
        s = th.sqrt(d).clamp_min(1e-12)
        norm = s.unsqueeze(-1) * s.unsqueeze(-2)
        corr = cov / norm
        return self.symmetrize(corr)

    def inner_product(self, A: th.Tensor, B: th.Tensor) -> th.Tensor:
        return th.einsum('...ij,...ij->...', A, B)


class CorPolyHyperbolicCholeskyMetric(Correlation):
    """
    C ↔ product of Poincaré balls via Cholesky + (hemisphere ↔ ball) row-wise.
    Public API:
      - to_poly_blocks(C):  [x1,...,x_{n-1}],  x_i ∈ ℝ^i
      - from_poly_blocks(xs): C
      - to_poly_ball(C):    concat coords z ∈ ℝ^{n(n-1)/2}
      - from_poly_ball(z):  C
      - projx(C):           project to Corr_n (SPD + diag=1)
    """
    def __init__(self, n: int, jitter: float = 1e-6):
        super().__init__(n=n, jitter=jitter)
        self.pball = Stereographic(K=-1.0)

    def projx(self, C: th.Tensor) -> th.Tensor:
        """Symmetrize → eig clip (λ≥jitter) → normalize to correlation."""
        C = self.symmetrize(C)
        evals, evecs = th.linalg.eigh(C)
        evals = evals.clamp_min(self.jitter)
        C_spd = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
        return self.covariance_to_correlation(C_spd)

    def to_poly_blocks(self, C: th.Tensor) -> List[th.Tensor]:
        """C → [x1,...,x_{n-1}], with x_i on the Poincaré ball B^i."""
        L = th.linalg.cholesky(C)  # expect valid correlation input
        xs: List[th.Tensor] = []
        for i in range(1, self.n):
            row = L[..., i, : i + 1]      # hemisphere in ℝ^{i+1}
            xi = hemisphere_to_poincare(row)  # → B^i
            xs.append(xi)
        return xs

    def from_poly_blocks(self, xs: List[th.Tensor]) -> th.Tensor:
        """[x1,...,x_{n-1}] → C."""
        n = self.n
        assert len(xs) == n - 1, f"Need {n-1} factors, got {len(xs)}."
        *batch, last = xs[-1].shape
        assert last == n - 1, "Last factor must have dim n-1."
        L = th.zeros(*batch, n, n, dtype=xs[0].dtype, device=xs[0].device)
        L[..., 0, 0] = 1.0
        for i, xi in enumerate(xs, start=1):
            vi = poincare_to_hemisphere(xi)  # ℝ^{i+1} hemisphere
            L[..., i, : i + 1] = vi
        C = L @ L.transpose(-1, -2)
        return self.projx(C)

    def correlation_to_poincare_concate(self, C):
        """PPBCM: Cor^+(n) \rightarrow \prod_{i=1}^{n-1} \pball{i},
                  \phi_{\hs{n} \rightarrow \pball{n}} \circ \Chol,
            input: [bs,...,n,n] correlation
            output: [bs, dim2], dim2 = \prod ... \times dim, with dim = n (n-1) /2
        """
        # Step 1: Perform Cholesky decomposition on C to get L
        L = th.linalg.cholesky(C)

        # Step 2: Map the lower triangular part of each row of L (from the 2nd row onward) to the Poincaré ball
        size = L.size()
        product_dims = th.prod(th.tensor(size[1:-2])).item()  # Product of all dims in "..."
        dim_in = int(product_dims * size[-1] * (size[-1] - 1) / 2)  # dim_in = product_dims * n * (n-1) / 2
        beta_n = beta(dim_in / 2, 1 / 2)
        mapped_rows = []
        for i in range(1, L.shape[-2]):  # Skip the first row (index 0)
            hs_r = L[..., i, :i+1]  # Take the first i+1 elements of row i
            pball_r = hemisphere_to_poincare(hs_r)  # Apply the hemisphere-to-Poincare transformation
            # Poincare beta concate
            beta_ni = beta(i / 2, 1 / 2)
            v_r = self.pball.log0(pball_r) * beta_n / beta_ni
            mapped_rows.append(v_r)

        x = self.pball.exp0(th.cat(mapped_rows, dim=-1).contiguous().view(size[0], -1)) # Shape will be [bs, dim_in]
        return x

    def poincare_concate_to_correlation(self, x: th.Tensor, c: int, n: int):
        """
        Convert from the \beta-concatenated Poincaré balls to a correlation matrix by constructing
        the Cholesky factor L using hemisphere transformations.

        Parameters:
        - x: Tensor of shape [bs, dim], where `dim` is the total feature dimension,
             representing the \beta-concatenated Poincaré balls.
        - c: Integer specifying the number of channels.
        - n: Integer specifying the total number of rows in the Cholesky factor L.
        - ball: Poincaré ball manifold object with methods log0 and exp0.

        Returns:
        - Tensor of shape [bs, c, n, n], representing the correlation matrix obtained from L * L^T.
        """
        bs, dim = x.shape
        expected_dim = c * sum(i for i in range(1, n))  # Sum of c * 1 + c * 2 + ... + c * (n-1)
        assert dim == expected_dim, f"Expected input dimension to be {expected_dim}, but got {dim}"

        x_tangent = self.pball.log0(x).contiguous().view(bs, c, -1)  # Map and Reshape to [bs, c, -1]
        beta_n = beta(dim / 2.0, 0.5)  # Compute the Beta function value B(dim/2, 0.5)

        # Initialize the Cholesky matrix L with zeros
        L = th.zeros((bs, c, n, n), dtype=x.dtype, device=x.device)
        L[..., 0, 0] = 1.0  # Set the first row of L as [1, 0, ..., 0]

        # Step 2: Process each segment with lengths 1, 2, ..., n-1 and reconstruct the rows of L
        current_idx = 0
        for i in range(1, n):
            segment_length = i  # Length of the i-th segment
            segment = x_tangent[..., current_idx:current_idx + segment_length]  # Extract the segment
            # n_segment = segment.numpy()
            # n_x_tangent = x_tangent.numpy()

            # Calculate β scaling factors for the segment
            beta_ni = beta(i / 2.0, 0.5)
            scaling_factor = beta_ni / beta_n

            # Scale and map the segment to hemisphere
            scaled_segment = segment * scaling_factor
            scaled_split = self.pball.exp0(scaled_segment)
            ith_hs = poincare_to_hemisphere(scaled_split)  # Shape: [bs, c, i]

            # Place the hemisphere-transformed segment in the lower triangular part of L
            L[..., i, :i + 1] = ith_hs  # Place the i-th segment in row `i`

            # Move to the next segment
            current_idx += segment_length

        # Step 3: Compute the correlation matrix C as C = L * L^T
        C = th.matmul(L, L.transpose(-1, -2))
        return C

    def poly_blocks_to_poincare_concate(self, xs: List[th.Tensor]) -> th.Tensor:
        """
        Input: xs = [x1,...,x_{n-1}], with x_i ∈ B^i, shape [..., i]
        Output: z ∈ B^D, shape [..., D], where D = sum_{i=1}^{n-1} i
        Rule: v_i = log0(x_i) * B(D/2, 1/2) / B(i/2, 1/2);
              v = concat_i v_i;
              z = exp0(v)
        """
        assert len(xs) >= 1, "xs cannot be empty"
        # Check that the last dimension is 1,2,...,k in ascending order
        for i, xi in enumerate(xs, start=1):
            assert xi.shape[-1] == i, f"Block {i} should have last-dim {i}, got {xi.shape[-1]}"

        D = sum(xi.shape[-1] for xi in xs)
        ref = xs[0]
        beta_D = th.as_tensor(float(beta(D / 2.0, 0.5)), dtype=ref.dtype, device=ref.device)

        vs = []
        for i, xi in enumerate(xs, start=1):
            vi = self.pball.log0(xi)  # [..., i]
            beta_i = th.as_tensor(float(beta(i / 2.0, 0.5)), dtype=xi.dtype, device=xi.device)
            vi = vi * (beta_D / beta_i)
            vs.append(vi)

        v_all = th.cat(vs, dim=-1)  # [..., D]
        z = self.pball.exp0(v_all)  # [..., D] in B^D
        return z

    def poincare_concate_to_poly_blocks(self, z: th.Tensor, n: Optional[int] = None) -> List[th.Tensor]:
        """
        Input: z ∈ B^D, shape [..., D], with D = n(n-1)/2
        Output: xs = [x1,...,x_{n-1}], each x_i ∈ B^i
        Rule: v = log0(z);
              v_i = segment(v, i) * B(i/2, 1/2) / B(D/2, 1/2);
              x_i = exp0(v_i)
        """
        if n is None:
            n = self.n
        D_expected = n * (n - 1) // 2
        assert z.shape[-1] == D_expected, f"Expected last-dim {D_expected}, got {z.shape[-1]}"

        v = self.pball.log0(z)  # [..., D]
        beta_D = th.as_tensor(float(beta(D_expected / 2.0, 0.5)), dtype=z.dtype, device=z.device)

        xs: List[th.Tensor] = []
        ofs = 0
        for i in range(1, n):
            seg = v[..., ofs:ofs + i]  # [..., i]
            ofs += i
            beta_i = th.as_tensor(float(beta(i / 2.0, 0.5)), dtype=z.dtype, device=z.device)
            seg_scaled = seg * (beta_i / beta_D)
            xi = self.pball.exp0(seg_scaled)  # [..., i] in B^i
            xs.append(xi)
        return xs




