import torch as th
from scipy.special import beta
from geoopt.manifolds import PoincareBall

from ..base import Manifold
from .ppbcm_functionals import hemisphere_to_poincare,poincare_to_hemisphere


cor_metrics = {'ECM','LECM','OLM','LSM'}

class Correlation(Manifold):
    """Computation for Correlation data with [...,n,n]"""
    def __init__(self, n):
        super().__init__()
        self.n=n; self.dim = int(n * (n - 1) / 2)
        self.register_buffer('I', th.eye(n))
    def _check_point_on_manifold(self, matrix, tol=1e-6):
        """
        Check if a batch of matrices are valid correlation matrices and provide detailed feedback.

        Parameters:
        - matrix: Input tensor of shape [..., n, n]
        - tol: Tolerance for floating point comparison

        Returns:
        - True if all matrices in the batch are valid correlation matrices, False otherwise
        """

        # Ensure matrix is at least 2D and square
        if matrix.shape[-1] != matrix.shape[-2]:
            print("Failed: Matrices must be square.")
            return False

        # Get matrix size n
        n = matrix.shape[-1]

        # 1. Check symmetry in batch (matrix should be equal to its transpose)
        if not th.allclose(matrix, matrix.transpose(-2, -1), atol=tol):
            print("Failed: Batch contains non-symmetric matrices.")
            return False

        # 2. Check positive semi-definiteness by checking if eigenvalues are non-negative
        eigenvalues = th.linalg.eigvalsh(matrix)  # Calculate eigenvalues for symmetric matrices
        if not th.all(eigenvalues >= -tol):
            print("Failed: Batch contains non-SPD (non-positive semi-definite) matrices.")
            return False

        # 3. Perform Cholesky decomposition to ensure positive definiteness
        try:
            L = th.linalg.cholesky(matrix)
        except RuntimeError:
            print("Failed: Batch contains matrices that are not positive definite (Cholesky decomposition failed).")
            return False

        # 4. Check that the diagonal elements of L are positive
        if not th.all(th.diagonal(L, dim1=-2, dim2=-1) > 0):
            print("Failed: Batch contains matrices with non-positive diagonal elements in Cholesky factor.")
            return False

        # 5. Check that each row of L has unit norm
        row_norms = th.sum(L ** 2, dim=-1)
        if not th.allclose(row_norms, th.ones_like(row_norms), atol=tol):
            print("Failed: Batch contains matrices whose Cholesky factor rows do not have unit norm.")
            return False

        print("Passed: All matrices are valid correlation matrices.")
        return True

    def symmetrize(self,X):
        return (X+X.transpose(-1,-2))/2

    def random(self,*shape,eps=1e-6):
        """ Generate random SPD matrices based on the given shape [..., n, n]."""
        assert len(shape) >= 2 and shape[-2] == shape[-1], "Shape must be [..., n, n] for square matrices"
        n = shape[-1]
        A = th.randn(shape)* 2 - 1
        spd_matrices = th.matmul(A, A.transpose(-2, -1)) + eps * th.eye(n, device=A.device)

        return self.covariance_to_correlation(spd_matrices)
    def covariance_to_correlation(self,cov_matrices):
        # Extract the diagonal elements (variances) from the covariance matrix
        diag_elements = th.diagonal(cov_matrices, dim1=-2, dim2=-1)
        # Compute the standard deviations (sqrt of the diagonal elements)
        std_devs = th.sqrt(diag_elements)
        # Outer product of standard deviations to form the normalization matrix
        normalization_matrix = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
        # Avoid division by zero in case of any zero variances (though variances are typically positive)
        normalization_matrix = th.where(normalization_matrix == 0, th.ones_like(normalization_matrix),
                                        normalization_matrix)
        # Compute the correlation matrix by dividing element-wise
        correlation_matrices = cov_matrices / normalization_matrix
        return correlation_matrices

    def inner_product(slef, A, B):
        # Ensure A and B are of the same shape [..., n, n]
        return th.einsum('...ij,...ij->...', A, B)

class CorPolyPoincareBallCholeskyMetric(Correlation):
    def __init__(self, n):
        super(__class__, self).__init__(n)
        self.pball=PoincareBall(c=1.0, learnable=False)
        self.register_buffer('c', th.tensor(1.0))

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
            v_r = self.pball.logmap0(pball_r) * beta_n / beta_ni
            mapped_rows.append(v_r)

        x = self.pball.expmap0(th.cat(mapped_rows, dim=-1).contiguous().view(size[0], -1)) # Shape will be [bs, dim_in]
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
        - ball: Poincaré ball manifold object with methods logmap0 and expmap0.

        Returns:
        - Tensor of shape [bs, c, n, n], representing the correlation matrix obtained from L * L^T.
        """
        bs, dim = x.shape
        expected_dim = c * sum(i for i in range(1, n))  # Sum of c * 1 + c * 2 + ... + c * (n-1)
        assert dim == expected_dim, f"Expected input dimension to be {expected_dim}, but got {dim}"

        x_tangent = self.pball.logmap0(x).contiguous().view(bs, c, -1)  # Map and Reshape to [bs, c, -1]
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
            scaled_split = self.pball.expmap0(scaled_segment)
            ith_hs = poincare_to_hemisphere(scaled_split)  # Shape: [bs, c, i]

            # Place the hemisphere-transformed segment in the lower triangular part of L
            L[..., i, :i + 1] = ith_hs  # Place the i-th segment in row `i`

            # Move to the next segment
            current_idx += segment_length

        # Step 3: Compute the correlation matrix C as C = L * L^T
        C = th.matmul(L, L.transpose(-1, -2))
        return C





