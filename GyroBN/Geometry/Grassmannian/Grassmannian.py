"""
    Author: Ziheng Chen
    Implementation of ONB perspective of the Grassmannian used in
    @inproceedings{chen2025gyrogroup,
        title={Gyrogroup Batch Normalization},
        author={Ziheng Chen and Yue Song and Xiaojun Wu and Nicu Sebe},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025}
    }
"""
import torch
import torch as th
from typing import Optional, Tuple, Union
from .utilities import aux_svd_logmap,gr_identity_batch

from GyroBN.Geometry import Manifold
from GyroBN.Geometry.base import EPS, TOLEPS

class Grassmannian(Manifold):
    """
    ONB perspective Grassmannian: [...,n,p]:
        A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects
        Gyrogroup Batch Normalization
        Riemannian Batch Normalization: A Gyro Approach
    """
    def __init__(self, n,p,exp_mode='cayley'):
        #exp_mode = cayley,expm
        super().__init__()
        self.n=n;self.p=p;
        self.register_buffer('identity', gr_identity_batch(n, p))
        self.register_buffer('In', th.eye(n))
        self.exp_mode=exp_mode
        self.is_vector_manifold = False

    def _check_point_on_manifold(
            self, x: th.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        # Calculate the dot product of x and its transpose, expecting an identity matrix for orthonormal columns
        x_dot_xT = th.matmul(x.transpose(-2, -1), x)
        identity = th.eye(x.shape[-1], device=x.device, dtype=x.dtype)

        # Check if the result is close to the identity matrix
        if th.allclose(x_dot_xT, identity, atol=atol, rtol=rtol):
            return True
        else:
            reason = "The columns of the input matrix are not orthonormal."
            return False, reason

    def _check_vector_on_tangent(
            self, x: th.Tensor, u: th.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """x^\top u = 0"""
        orthogonality_condition = th.matmul(x.transpose(-2, -1), u)
        if th.allclose(orthogonality_condition, th.zeros_like(orthogonality_condition), atol=atol, rtol=rtol):
            return True, None
        else:
            return False, "At least one vector does not satisfy the orthogonality condition within the given tolerances."

    def random_tangent_origin(self, *size, mean=0, std=1, scale=1, full_matrix=False):
        r"""
        Sample a Gaussian tangent at the ONB origin {I}_{n,p} on Gr(n,p).

        - size: (..., n, p)  # last two dims must be (self.n, self.p)
        - Tangent block form at origin: Δ = [0_{p×p}; B_{(n-p)×p}],
          with B ~ Normal(mean, (std)^2) * scale.
        - Returns B (shape: (..., n-p, p)) if full_matrix=False, else Δ (shape: (..., n, p)).
        """
        if len(size) < 2:
            raise ValueError("size must include (..., n, p) as the last two dimensions.")

        n_req, p_req = size[-2], size[-1]
        n, p = self.n, self.p
        if (n_req != n) or (p_req != p):
            raise ValueError(f"size mismatch: got n={n_req}, p={p_req} but manifold has n={n}, p={p}.")
        batch = size[:-2]
        B = (th.randn(*batch, n - p, p) * std + mean) * scale
        if full_matrix:
            top_zero = th.zeros(*batch, p, p, device=B.device, dtype=B.dtype)
            return th.cat([top_zero, B], dim=-2)  # Δ with shape (*size, n, p)
        else:
            return B  # shape (*size, n-p, p)

    def qr_flipped(self, X):
        """ Compute qr with flipping, which might be useful for standard Riem log."""
        Q, R = th.linalg.qr(X)
        # flipping
        output = th.matmul(Q, th.diag_embed(th.sign(th.sign(th.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return output

    def rand(self, *size):
        """ Generates a random point on the Grassmannian manifold with dimensions [..., n, p]. """
        *leading_dims, n, p = size
        # Ensure p is not greater than n
        if p >= n:
            raise ValueError("p should not be greater than n.")
        shape = (*leading_dims, n, p) if leading_dims else (n, p)
        random_matrix = th.rand(*shape)
        q, _ = th.linalg.qr(random_matrix)
        return q

    def zero(self, *shape):
        return gr_identity_batch(shape)

    def zero_tan(self, *shape):
        return th.zeros(*shape)

    def zero_tan_like(self, x: th.Tensor):
        return th.zeros_like(x)

    def zero_like(self, x):
        return gr_identity_batch(x.shape,dtype=x.dtype, device=x.device)

    def dist(self,x, y, squared: bool = False, keepdim = False):
        _,S,_ = th.linalg.svd(x.transpose(-1,-2).matmul(y))
        # Clamp S to the range [-1, 1] for numerical stability
        S_clamped = th.clamp(S, -1, 1)
        principal_angles = th.acos(S_clamped)

        if squared:
            res = torch.sum(principal_angles ** 2, dim=-1)
        else:
            res = torch.norm(principal_angles, dim=-1)
        if keepdim:
            # ensure shape [..., 1, 1]
            res = res.unsqueeze(-1).unsqueeze(-1)
        return res

    def dist0(self, U, squared: bool = False, keepdim = False):
        *batch_shape, n, p = U.shape

        # Calculate I_{n,p}^T * U, this simplifies to selecting the top block of U
        U_1 = U[..., :p, :]  # Shape: [..., p, p]

        # Calculate the geodesic distance using the principal angles
        _, S, _ = th.linalg.svd(U_1)
        S_clamped = th.clamp(S, min=0, max=1)
        principal_angles = th.acos(S_clamped)

        if squared:
            res = torch.sum(principal_angles ** 2, dim=-1)
        else:
            res = torch.linalg.vector_norm(principal_angles, ord=2, dim=-1)
        if keepdim:
            # ensure shape [..., 1, 1]
            res = res.unsqueeze(-1).unsqueeze(-1)

        return res

    def exp(self, U0, Delta, project=True):
        """
        Compute the Grassmann exponential \rieexp_{U0}(Delta).
        """
        Q, Sigma, Vh = th.linalg.svd(Delta,full_matrices=False)
        cosSigma = th.cos(Sigma).unsqueeze(-2)
        sinSigma = th.sin(Sigma).unsqueeze(-2)
        U1 = (U0.matmul(Vh.transpose(-2, -1)).mul(cosSigma) + Q.mul(sinSigma)).matmul(Vh)
        # Our exp on log_exp indicates this is minor. But manopt indicates that re-orth might be important, possiblily for optimization
        res = self.qr_flipped(U1) if project else U1
        return res

    def exp0(self, Delta, project=True):
        """ \rieexp_{\idonb}(Delta)."""
        # Extract the (n-p)×p block B
        if Delta.shape[-2] == self.n:  # full tangent [0; B]
            B = Delta[..., self.p:, :]
        else:  # already B
            B = Delta

        Q, Sigma, Vh = th.linalg.svd(B, full_matrices=False)
        R = Vh.transpose(-2, -1)
        cosS = th.cos(Sigma).unsqueeze(-2)  # (..., 1, p)
        sinS = th.sin(Sigma).unsqueeze(-2)  # (..., 1, p)
        # Top: U1 = R cosΣ R^T   (R diag(cosS) R^T)
        U1 = (R * cosS) @ R.transpose(-2, -1)
        # Bottom: U2 = Q sinΣ R^T (Q diag(sinS) R^T)
        U2 = (Q * sinS) @ R.transpose(-2, -1)
        U = th.cat([U1, U2], dim=-2)
        res = self.qr_flipped(U) if project else U
        return res

    def log_standard(self,x, y):
        """
            Perform a logarithmic map :math:`\operatorname{Log}_{x}(y)`.
            Note that z=self.expmap(x,self.log(x,y)), then z != y, but self.dist(z,y) is almost 0
        """
        ytx = y.transpose(-1, -2).matmul(x)
        At = y.transpose(-1, -2).subtract(ytx.matmul(x.transpose(-1, -2)))
        Bt = th.linalg.pinv(ytx).matmul(At)
        # Bt = th.linalg.solve(ytx,At)
        u, s, vh = th.linalg.svd(Bt.transpose(-1, -2), full_matrices=False)
        s_atan = th.atan(s)

        return u.mul(s_atan.unsqueeze(-2)).matmul(vh)

    def log(self, U0, U1, is_return_tuple=False):
        """
        Compute the Grassmann Riemannian logarithm \riemlog_{U0}(U1) by single svd [Alg. 5.3, Bendokat,2024].
        Parameters:
            U0, U1: Stiefel representatives of subspaces, shape = [..., n, p]
            is_return_tuple: needed for efficient geodesic
        intermediate:
            U1star: Adapted Stiefel representative of U1
        Returns:
            Delta: Tangent vector in horizontal space at U0, from U0 to U1star
            or U, arctan(\Sigma), Vh with [...,n,p], [...,p] and [...,n,p]
        Note that
            1. Different from the offcicial Matlab code, we solve several singular cases for the forward computation
            2. U might fail to be orthonormal, if arctan(\Sigma_i) = 0, but this do not affect the computation of geodeisc
            3. This code might fail to deal with the BP under U0=U1, but we don't need it.
            4. We deal with the BP of the case of (\bar{\Sigma_i}) = 0
            5. In the calculation of geodesic, arctan(\Sigma_i) = 0 might affect BP of singvals, but we don't need it
        """
        # Check if U1 and U0 are essentially the same
        if th.allclose(U0, U1, atol=EPS[U0.dtype]):
            # If singvals is zero, then U1 and U0 are the same and Delta should be zero
            # this case might undermine the auto differentiation, but we do not need to care about this issue currently.
            if is_return_tuple:
                n, p = U1.shape[-2], U1.shape[-1]
                asin_singvals = th.zeros_like(U1)[..., -1, :] if len(U1.shape) > 2 else th.zeros_like(U1[-1,:])
                Q_hat = th.eye(n, p,dtype=U0.dtype,device=U0.device)
                R1_vh = th.eye(p, p,dtype=U0.dtype,device=U0.device)
                return Q_hat, asin_singvals, R1_vh
            else:
                # note that we use U1 here, as U0 might be [c,n,p]
                return th.zeros_like(U1)
        else:
            # Step 1: Procrustes, svd is efficiently calculated on q \times q matrices
            Q1_ascending, S1_ascending, R1_vh_ascending = aux_svd_logmap(th.matmul(U1.transpose(-2, -1), U0))

            # Calculate new rep
            U1star = th.matmul(U1, Q1_ascending)
            # Step 3: SVD without actual SVD
            H = U1star - th.matmul(U0, th.matmul(U0.transpose(-2, -1), U1star))
            singvals = th.sqrt(1 - S1_ascending ** 2)
            # this is the sigma of \delta_U0
            asin_singvals = th.asin(singvals)

            # Resolve 0/0 ambiguity by ensuring orthogonality of Q2
            # note  that it is okay for singvals approching 0, but cannot be 0
            # RHS to make sure when 0/0 happens, we should have \partial{L} / \partial{sigma_i} = \partial{L} / \partial{th.asin(singvals)}
            asin_div_singvals = th.where(singvals != 0,  asin_singvals/ singvals, asin_singvals)
            # Step 3: Return tuple or tangent vector
            if is_return_tuple:
                singvals_expanded = singvals.unsqueeze(-2)  # Adds a new dimension to match H's shape for division
                condition = singvals_expanded != 0
                # Note that in geodesic, this might affect BP of singvals, which is not our case
                Q_hat = th.where(condition, H.div(singvals_expanded), H)
                return Q_hat, asin_singvals, R1_vh_ascending
            else:
                Delta = th.matmul(th.mul(H, asin_div_singvals.unsqueeze(-2)), R1_vh_ascending)
                return Delta

    def log0(self, U, is_lower_part=False):
        """Efficient computation of \riemlog_{I_{n,p}}(U) by single svd, reducing the p \times p inverse and n \times p SVD into a single p \times p SVD:
            Parameters:
                U: Stiefel representatives of subspaces, shape = [..., n, p]
            Returns:
                Delta: Tangent vector in horizontal space at I_{n,p}, shape = [..., n, p]
            Note that we deal with the follwoing singular case w.r.t. BP
                1. When U =I_{n,p}
                2. When sigma_i = 0 in [Alg. 5.3, Bendokat,2024].
        """
        if th.allclose(U, self.identity, atol=EPS[U.dtype], rtol=EPS[U.dtype]):
            # this is for BP, as Log_{P*,P} = Id
            Delta = U - U.detach()
            if is_lower_part:
                # If only the lower part is needed, slice accordingly
                return Delta[..., U.shape[-1]:, :]
            else:
                return Delta
        else:
            # Step 1: svd is efficiently calculated on p \times p upper part of U
            n, p = U.shape[-2], U.shape[-1]  # Last two dimensions sizes are n and p respectively
            U_upper = U[..., :p, :]
            U_lower = U[..., p:, :]
            Q1_ascending, S1_ascending, R1_vh_ascending = aux_svd_logmap(U_upper.transpose(-1,-2))
            # Step 2: calculate lower part
            singvals = th.sqrt(1 - S1_ascending ** 2)
            asin_singvals = th.asin(singvals)
            Sigma_div_singvals = th.where(singvals != 0, asin_singvals / singvals, asin_singvals)
            U1star = th.matmul(U_lower, Q1_ascending)
            Delta_lower = th.matmul(th.mul(U1star, Sigma_div_singvals.unsqueeze(-2)), R1_vh_ascending)
            # Step 3: concatenation
            if is_lower_part:
                return Delta_lower
            else:
                zeros_upper = th.zeros(*U.shape[:-2], p, p, dtype=U.dtype, device=U.device)
                Delta = th.cat([zeros_upper, Delta_lower], dim=-2)
                return Delta

    def geodesic(self, U0, U1,t):
        """ The geodesic from U0 to U1 with single svd"""
        #Note that we don't re-orth Q_i when arctan(\Sigma_i) = 0, but this does not affect the computation of geodesic
        #this might cause problem in BP, but we don't need it
        Q, simga,Vh = self.log(U0,U1,is_return_tuple=True)
        singma_new = t * simga
        costSigma = th.cos(singma_new).unsqueeze(-2)
        sintSigma = th.sin(singma_new).unsqueeze(-2)
        U_new = (U0.matmul(Vh.transpose(-2, -1)).mul(costSigma) + Q.mul(sintSigma)).matmul(Vh)
        return U_new

    def transpfrom0(self,P,B,zero):
        """Parallel transport delta from I_{n,p} to P,
         delta = [0,
                  B] \in T_{I_{n,p}}Gr"""

        if th.allclose(P, self.identity, atol=EPS[P.dtype], rtol=EPS[P.dtype]):
            delta_new = th.cat([zero, B], dim=-2)
        else:
            # Step 1: svd of H = Log_{I_{n,p}} Y
            n, p = P.shape[-2], P.shape[-1]  # Last two dimensions sizes are n and p respectively
            P_1 = P[..., :p, :]
            P_2 = P[..., p:, :]
            tmp = P_2 @ th.linalg.pinv(P_1)

            U, Sigma, Vh = th.linalg.svd(tmp,full_matrices=False)
            V=Vh.transpose(-1,-2)
            Sigma_atan = th.atan(Sigma)

            # Step 2: PT of delta of a shape like [0,B^\top] \in T_{I_{n,p}}Gr
            cosSigma_atan = th.cos(Sigma_atan).unsqueeze(-2)
            sinSigma_atan = th.sin(Sigma_atan).unsqueeze(-2)
            upper_part = -V.mul(sinSigma_atan) @ U.transpose(-1,-2)
            lower_part = U.mul(cosSigma_atan) @ U.transpose(-1, -2) + th.eye(n-p,dtype=B.dtype,device=B.device)- U @ U.transpose(-1,-2)
            T = th.cat([upper_part, lower_part], dim=-2)
            delta_new = T @ B

        return delta_new

    def PP2ONB(self,P):
        S, U = th.linalg.eigh(P)
        S_desc, indices = th.sort(S, descending=True)

        # Sort eigenvectors to correspond to sorted eigenvalues
        U_desc = th.gather(U, -1, indices.unsqueeze(-2).expand_as(U))

        output = U_desc[..., :self.p]

        return output

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ONB (Stiefel) parallel transport of v ∈ T_x Gr(n,p) to T_y Gr(n,p)
        along the minimal geodesic from x to y.

        Uses the Edelman–Arias–Smith closed form (Eq. 2.68), reusing the tuple
        (U, Θ, V^T) returned by self.log(x, y, is_return_tuple=True).

        Complexity: O(n p^2). Numerically projects the result back to T_y.

        References:
          - Edelman, Arias, Smith (1998), §2.5.2 Thm. 2.4 (Eq. 2.68) and Eq. (3.2).  # see paper
        """
        # Quick exit: identical endpoints → identity transport (project for safety).
        tol = EPS[x.dtype]
        if torch.allclose(x, y, atol=tol, rtol=tol):
            return self.proju(y, v)

        # Ensure v is tangent at x
        v = self.proju(x, v)

        # Geodesic data from our single-SVD log (Q_hat=U, asin_singvals=Θ, Vh=V^T)
        U, theta, Vh = self.log(x, y, is_return_tuple=True)  # shapes: (..., n, p), (..., p), (..., p, p)
        V = Vh.transpose(-2, -1)  # (..., p, p)

        # Precompute trigs and small blocks
        sinT = torch.sin(theta).unsqueeze(-2)  # (..., 1, p)
        cosT = torch.cos(theta).unsqueeze(-2)  # (..., 1, p)

        YV = x @ V  # (..., n, p)
        UTv = U.transpose(-2, -1) @ v  # (..., p, p)

        # Correction term:  (x V sinΘ + U (1 - cosΘ)) (U^T v)
        correction = (YV * sinT) + (U * (1.0 - cosT))  # (..., n, p)
        transported = v - correction @ UTv  # (..., n, p)

        # Clean up roundoff: ensure result is horizontal at y
        return self.proju(y, transported)

    #----- The following is needed for optimization. We basically reimplement the manopt into torch. --------
    def egrad2rgrad(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.proju(x,u)

    def inner(
            self, x: th.Tensor, u: th.Tensor, v: th.Tensor = None, *, keepdim=False
    ) -> th.Tensor:
        if v is None:
            v = u
        return th.sum(u * v, dim=(-2, -1), keepdim=keepdim)

    def inner0(self, u: torch.Tensor, v: torch.Tensor = None, *, keepdim: bool = False) -> torch.Tensor:
        """
        Canonical inner product at the ONB identity I_{p,n}.
        u,v are ambient [..., n, p] tangent candidates; only their lower (n-p)×p blocks matter.
        """
        if v is None:
            v = u
        # Extract lower blocks
        u_low = u[..., self.p:, :]  # shape [..., n-p, p]
        v_low = v[..., self.p:, :]
        # Frobenius inner product
        res = torch.sum(u_low * v_low, dim=(-2, -1), keepdim=keepdim)
        return res

    def proju(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        """Orthogonal projection of an ambient vector U to the horizontal space at X."""
        xtu = th.matmul(x.transpose(-2, -1), u)
        tangent_vec = u - th.matmul(x, xtu)
        return tangent_vec

    def proju0(self, u: th.Tensor) -> th.Tensor:
        """
        return: zero the top p×p block, keep the lower (n-p)×p block.
        """
        mask = torch.zeros(self.n, self.p, dtype=u.dtype, device=u.device)
        mask[self.p:, :] = 1.0
        return u * mask

    def projx(self, x: th.Tensor) -> th.Tensor:
        # Perform QR decomposition on x. Following manopt, we do not need to worry about flipping signs of columns here
        q, _ = th.linalg.qr(x)
        return q

    def retr(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        # see manopt for the reason of polar over QR
        # Compute the polar factorization of y = x + u.
        u, _, vt = th.linalg.svd(x + u, full_matrices=False)
        # x_new = u @ vt
        return u @ vt

    def _transp(self, x: th.Tensor, y: th.Tensor, v: th.Tensor) -> th.Tensor:
        # vector transport for Grassmannian, which is used in optimization
        return self.proju(y, v)


