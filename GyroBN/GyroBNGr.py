import torch as th
import torch.nn as nn

from GyroBN import GyroBNBase
from .Geometry.Grassmannian.utilities import gr_identity_batch
from .Geometry.Grassmannian.GrGyro import GrassmannianGyro

class GyroBNGr(GyroBNBase):
    """
    GyroBN on the Grassmannian (ONB perspective).

    Args:
        shape (list[int]): [n, p] or [c, n, p].
        exp_mode (str, default='cayley'): 'cayley' or 'expm' for the map exponentiation.
        max_iter (int, default=1): Karcher/Fréchet mean iterations.
        batchdim (list[int], default=[0]): [0] or [0,1].
        momentum (float, default=0.1): Momentum for running statistics.
        track_running_stats (bool, default=True): Keep running mean/var.
        init_1st_batch (bool, default=False): Initialize running stats by the first batch.
        eps (float, default=1e-6): Numerical stability for variance.

    Note:
        - Input are expected to be [bs,c,n,p] or [bs,n,p]
        - Following [Nguyen 2023], Biasing parameter are parameterized by the exponential coordinate at \tilde{I}_{n,p},
        and we use cayley map for efficient implementation
        - Following SPDBN, TSMNet, and LieBN, we use detach for Fréchet mean and variance.

    References:
        - Ziheng Chen, et al, Gyrogroup Batch Normalization. ICLR 2024
        - Ziheng Chen, et al, Riemannian Batch Normalization: A Gyro Approach. 2025
        - Xuan Son Nguyen, et al, Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach. ICML 2023
        - Thomas Bendokat, et al, A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects. Adv. Comput. Math. 2024
        - Daniel Brooks, et al, Riemannian Batch Normalization for SPD Neural Networks. NeurIPS 2019
        - Reinmar Kobler, et al, SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG. NeurIPS 2022
        - Ziheng Chen, et al, A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024
    """
    def __init__(self, shape, exp_mode='cayley', max_iter=1,
                 batchdim = [0],momentum=0.1, track_running_stats=True, init_1st_batch=False,eps=1e-6):
        """
            shape : [c, n, p];
            exp_mode: cayley, expm;
        """
        super().__init__(shape=shape, batchdim=batchdim, momentum=momentum,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.max_iter, self.exp_mode = max_iter, exp_mode
        # --- validate shape & batchdim compatibility ---
        self._validate_shape_and_batchdim()

        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()
    def _validate_shape_and_batchdim(self):
        """Sanity checks for shape and batchdim compatibility."""
        if not isinstance(self.shape, (list, tuple, th.Size)):
            raise TypeError(f"`shape` must be a list/tuple/Size, got {type(self.shape)}")

        shape_list = [int(x) for x in self.shape]
        if len(shape_list) not in (2, 3):
            raise ValueError(f"`shape` must be [n,p] or [c,n,p], got {shape_list}")

        # basic geometry constraints
        n, p = int(shape_list[-2]), int(shape_list[-1])
        if not (1 <= p <= n):
            raise ValueError(f"Require 1 <= p <= n for Grassmannian, got n={n}, p={p}")
        if len(shape_list) == 3:
            c = int(shape_list[0])
            if c <= 0:
                raise ValueError(f"Channel dimension c must be positive, got c={c}")

        # batchdim constraints for your use cases
        if self.batchdim not in ([0], [0, 1]):
            raise ValueError(f"`batchdim` must be [0] or [0,1], got {self.batchdim}")

        # shape vs batchdim compatibility:
        # - shape=[n,p]  : only [0] is valid (no channel axis to reduce over)
        # - shape=[c,n,p]: both [0] (per-channel) and [0,1] (shared across channel) are valid
        if len(shape_list) == 2 and self.batchdim != [0]:
            raise ValueError(f"For shape=[n,p], batchdim must be [0], got {self.batchdim}")

        # exp_mode
        if self.exp_mode not in ('cayley', 'expm'):
            raise ValueError(f"`exp_mode` must be 'cayley' or 'expm', got {self.exp_mode}")

    def get_manifold(self):
        n, p = map(int, self.shape[-2:])
        self.manifold = GrassmannianGyro(n, p, exp_mode=self.exp_mode)

    def _get_param_shapes(self):
        """
        Shapes for parameters/statistics given shape=[n,p] or [c,n,p].
        Returns (weight_shape, mean_shape, var_shape) **with prefix included**:
          - mean  : prefix + [n, p]
          - var   : prefix + [1, 1] (or [1] when no prefix; still broadcasts)
          - weight: prefix + [n - p, p]  (tangent at \tilde{I}_{n,p})
        Note: Whether to use the prefix is decided by callers based on `batchdim`.
        """
        *shape_prefix, n, p = [int(x) for x in self.shape]
        mean_shape = shape_prefix + [n, p]
        var_shape = shape_prefix + [1, 1] if shape_prefix else [1]
        weight_shape = shape_prefix + [n - p, p]
        return weight_shape, mean_shape, var_shape

    def set_parameters(self):
        w_pref, _, v_pref = self._get_param_shapes()
        if len(self.batchdim) == 1:
            # per-channel/group parameters (use prefix)
            self.weight = nn.Parameter(th.zeros(*w_pref))  # [..., n-p, p]
            self.shift  = nn.Parameter(th.ones(*v_pref))   # [..., 1, 1] or [1]
        else:
            # shared parameters across channel (ignore prefix)
            n, p = map(int, self.shape[-2:])
            self.weight = nn.Parameter(th.zeros(n - p, p))  # [n-p, p]
            self.shift  = nn.Parameter(th.ones(1))          # [1]

    def register_running_stats(self):
        # Lazy init by first batch if requested
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var  = None
            return

        _, m_pref, v_pref = self._get_param_shapes()

        if len(self.batchdim) == 1:
            # per-channel/group running stats (use prefix)
            self.register_buffer('running_mean', gr_identity_batch(*m_pref))  # e.g., [c,n,p]
            self.register_buffer('running_var',  th.ones(*v_pref))            # e.g., [c,1,1]
        else:
            # shared running stats across channel (ignore prefix)
            self.register_buffer('running_mean', gr_identity_batch(m_pref[-2], m_pref[-1]))  # [n,p]
            self.register_buffer('running_var',  th.ones(1))

    def forward(self, X):
        if (self.training):
            # compute batch mean, following SPDNetBN, TSMNet, and LieBN we use detach.
            input_mean = self.manifold.frechet_mean(X.detach(),max_iter=self.max_iter, batchdim=self.batchdim, cond_dim=(-1,-2))
            input_var = self.manifold.frechet_variance(X.detach(),input_mean, batchdim=self.batchdim, is_unsqueeze=False)
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            input_mean = self.manifold.frechet_mean(X.detach(),max_iter=self.max_iter, batchdim=self.batchdim, cond_dim=(-1,-2))
            input_var = self.manifold.frechet_variance(X.detach(),input_mean, batchdim=self.batchdim, is_unsqueeze=False)

        output = self.normalization(X, input_mean, input_var)
        return output

    def normalization(self, X, input_mean, input_var):
        # centering
        X_centered = self.manifold.gyroadd(input_mean, X, is_inverse=True)
        # scaling and shifting
        factor = self.shift / (input_var+self.eps).sqrt()
        X_scaled = self.manifold.gyroscalarprod(factor, X_centered)
        # biasing
        weight = self.manifold.B2skrew(self.weight)
        X_normalized = self.manifold._gyroadd_skew2U(weight, X_scaled, is_inverse=False)
        return X_normalized



