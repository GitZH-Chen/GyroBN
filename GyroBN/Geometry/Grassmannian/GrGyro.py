"""
    Author: Ziheng Chen
    Implementation of ONB perspective of the Grassmannian gyrovector spaces used in
    @inproceedings{chen2025gyrogroup,
        title={Gyrogroup Batch Normalization},
        author={Ziheng Chen and Yue Song and Xiaojun Wu and Nicu Sebe},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025}
    }
"""

import torch as th
from .Grassmannian import Grassmannian

class GrassmannianGyro(Grassmannian):
    """
        ONB perspective of the Gyro Computation for Grassmannian data with size of [...,n,p]:
            Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach
            Gyrogroup Batch Normalization
            Riemannian Batch Normalization: A Gyro Approach
    """
    def __init__(self, n,p,exp_mode='cayley'):
        #exp_mode = cayley,expm
        super().__init__(n=n,p=p)
        self.register_buffer('In', th.eye(n))
        self.exp_mode=exp_mode

    def B2skrew(self, B):
        """
        Construct skrew symmetric matrtices from B, by Eq.(2.8) with Q=I_n in [Bendokat,2024]
        Input: [..., n-p, p]
        Returns a skew-symmetric matrix omega like
            omega[...,:,:] = [0,    -B^T]
                             [B,     0 ]
        """
        n = B.size(-1) + B.size(-2)
        p = B.size(-1)
        # Adjust the initialization of omega to accommodate any number of leading dimensions
        omega_shape = list(B.shape[:-2]) + [n, n]
        omega = th.zeros(omega_shape, dtype=B.dtype, device=B.device)
        # Place `-B^T` (the negative transpose of `B`) in the upper right block
        omega[..., :p, p:] = -B.transpose(-2, -1)
        # Place `B` in the lower left block
        omega[..., p:, :p] = B
        return omega

    def exp_skew(self, X):
        """matrix exponential: Skew \rightarrow SO(n)"""
        if self.exp_mode=='cayley':
            # cayley(X)=(I - X) @ (I + X)^{-1}
            X_new = th.linalg.solve(self.In + X, self.In - X, left=False)
        elif self.exp_mode=='expm':
            X_new = th.linalg.matrix_exp(X)
        else:
            raise NotImplementedError
        return X_new

    def exp_skew_blockB(self, B):
        """  reduce the n \times n expm into a n-p \times p svd
        expm for
            [0,    -B^T]
            [B,     0 ]
            B is [..., n-p, p], and assert n-p >= p
        """
        assert B.shape[-2] >= B.shape[-1]
        # Perform full singular value decomposition of B
        U, Sigma, Vh = th.linalg.svd(B, full_matrices=True)

        # Compute the cosine and sine of Sigma
        cos_Sigma = th.cos(Sigma)
        sin_Sigma = th.sin(Sigma)

        # Calculate the blocks
        V = Vh.transpose(-2, -1)
        W1 = U[..., :, :B.shape[-1]]  # [..., n-p x p]
        W2 = U[..., :, B.shape[-1]:]  # [..., n-p x (n- p - p)]

        # Perform element-wise multiplication for diagonal matrices
        upper_left_block = (V * cos_Sigma.unsqueeze(-2)) @ Vh
        upper_right_block = -(V * sin_Sigma.unsqueeze(-2)) @ W1.transpose(-2, -1)
        lower_left_block = (W1 * sin_Sigma.unsqueeze(-2)) @ Vh
        lower_right_block = (W1 * cos_Sigma.unsqueeze(-2)) @ W1.transpose(-2, -1) + W2 @ W2.transpose(-2, -1)

        # Concatenate the blocks to form the full matrix exponential
        upper_blocks = th.cat((upper_left_block, upper_right_block), dim=-1)
        lower_blocks = th.cat((lower_left_block, lower_right_block), dim=-1)
        exp_B = th.cat((upper_blocks, lower_blocks), dim=-2)

        return exp_B

    def exp0_pp(self, skew_a: th.Tensor) -> th.Tensor:
        """
        Exponential map at the identity: math:`Exp_{\tilde{I}_{n,p}}(skew_a)`.
            skew_a: [...,n,n] th.Tensor skew-symmetric matrices
        """
        tmp = th.matmul(self.exp_skew(skew_a), self.identity)
        return tmp.matmul(tmp.transpose(-1,-2))

    def get_omega(self,U: th.Tensor):
        """omega = [log_{\tilde{I}_{n,p} (\tilde{U}), \tilde{I}_{n,p}}] \in Skew{n} with \tilde{U}=U U^\top """
        # lower_part of logmap_id under OBN, needed by self.left_gyrotranslation_V2U and self.gyro_scalarproduct
        bar_U_2 = self.log0(U, is_lower_part=True)
        # [log_{\tilde{I}_{n,p} (\tilde{V}), \tilde{I}_{n,p}}]
        omega = self.B2skrew(bar_U_2)
        return omega
    def gyroadd(self, V: th.Tensor, U: th.Tensor, is_inverse: bool = False):
        """
        left gyro translation:
        input:
            V,U: [...,n,p] Stiefel representatives

        if is_inverse:
            (\ominus V) \oplus U = exp(-omega) U, with omega defined as
                [log_{\tilde{I}_{n,p} (\tilde{V}), \tilde{I}_{n,p}}] \in Skew{n} with \tilde{V}=V V^\top
        else:
            V \oplus U = exp(omega) U, with skew_a is
        """
        # [log_{\tilde{I}_{n,p} (\tilde{V}), \tilde{I}_{n,p}}]
        omega = self.get_omega(V)
        return self._gyroadd_skew2U(omega,U,is_inverse)

    def _gyroadd_skew2U(self, omega: th.Tensor, U: th.Tensor, is_inverse: bool =False):
        """ left gyro translation of exp(omega) U
            following https://arxiv.org/pdf/1909.09501.pdf, we use cayley map to approximate expm(\omega), where \omega \in Skew{n}
        """
        skew_a = -omega if is_inverse else omega
        orth= self.exp_skew(skew_a)
        U_new = orth.matmul(U)
        return U_new

    def gyroscalarprod(self, t, U: th.Tensor):
        """Gyro scalar product:
            t \odot U = exp(t*omega)I_{n,p}
            omega = [log_{\tilde{I}_{n,p} (\tilde{V}), \tilde{I}_{n,p}}] \in Skew{n} with \tilde{V}=V V^\top
        """
        omega = self.get_omega(U)
        orth = self.exp_skew(t*omega)
        return orth[..., :U.shape[-1]]

    def gyro_scalarproduct_ONB(self, t: float, U: th.Tensor):
        """ Gyro scalar product without expm(\omega) for n \times n skew matrices
            for the case of n-p<p, we should add orthonormal completion
        """
        n,p=U.shape[-2],U.shape[-1]
        raise NotImplementedError


