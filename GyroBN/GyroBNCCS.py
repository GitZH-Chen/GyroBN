import torch
import torch.nn as nn

from GyroBN import GyroBNBase
from .Geometry.constantcurvature import classes

class GyroBNCCS(GyroBNBase):
    """
    GyroBN for Constant Curvature Spaces (CCS): Stereographic / Hyperboloid / Klein / Sphere.

    Args:
        shape (list[int]): Manifold dimension (e.g., [d]).
        model (str, default='Stereographic'): One of {"Stereographic","Hyperboloid","Klein","Sphere"}.
        K (float, default=-1.0): Curvature (K>0 sphere; K=0 Euclidean; K<0 hyperbolic).
        max_iter (int, default=1000): Max iterations for Fréchet mean.
        momentum (float, default=0.1): Momentum for running statistics.
        track_running_stats (bool, default=True): Keep running mean/var.
        init_1st_batch (bool, default=False): Initialize running stats by the first batch.
        eps (float, default=1e-6): Numerical stability for variance.
        clamp_factor (float|None, default=None): Optional upper bound for the scaling factor.

    Note:
        Input expected shape: [batch_size, dim].

    References:
    - Ziheng Chen, et al, Gyrogroup Batch Normalization. ICLR 2024
    - Ziheng Chen, et al, Riemannian Batch Normalization: A Gyro Approach. 2025
    - Gregor Bachmann, et al, Constant Curvature Graph Convolutional Networks. ICML 2020
    - Ondrej Skopek, et al, Mixed-Curvature Variational Autoencoders. ICLR 2020
    - Ungar, A.A. Analytic Hyperbolic Geometry and Albert Einstein’s Special Theory of Relativity (2nd ed.). World Scientific, 2022
    """

    def __init__(self, shape, model='Stereographic', K=-1.0, max_iter=1000,
                 momentum=0.1, track_running_stats=True, init_1st_batch=False,eps=1e-6,clamp_factor=None):
        super().__init__(shape=shape, batchdim=[0], momentum=momentum,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.model = model;self.K=K;self.max_iter=max_iter
        self.clamp_factor=clamp_factor
        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()

    def get_manifold(self):
        self.manifold = classes[self.model](K=self.K)

    def _get_param_shapes(self):
        shape_prefix = list(self.shape[:-1])
        mean_shape = shape_prefix + [self.manifold.dim_to_sh(self.shape[-1])]
        var_shape = shape_prefix + [1] if shape_prefix else [1]
        return mean_shape, var_shape

    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.shift = nn.Parameter(torch.ones(*var_shape))
        self.weight = nn.Parameter(self.manifold.zero_tan(mean_shape))

    def register_running_stats(self):
        """
        In [a], they use the following for the variance:
            Initialization:
                self.updates = 0
            updates:
                self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates
                self.updates += 1
        But we follow the standard Euclidean BN.
        [a] Differentiating through the Fréchet Mean
        """
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = self.manifold.zero(mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)

    def forward(self, x):
        if self.training:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
        output = self.normalization(x,input_mean,input_var)
        return output

    def normalization(self,x,input_mean,input_var):
        # centering
        inv_input_mean = self.manifold.gyroinv(input_mean)
        x_center = self.manifold.gyrotrans(inv_input_mean,x)
        # shifting
        factor = self.shift / (input_var + self.eps).sqrt()
        if self.clamp_factor is not None:
            factor = factor.clamp(max=self.clamp_factor)
        x_scaled = self.manifold.gyroscalarprod(x_center,factor)
        # biasing
        weight = self.manifold.exp0(self.weight)
        x_normed = self.manifold.gyrotrans(weight, x_scaled)
        return x_normed

