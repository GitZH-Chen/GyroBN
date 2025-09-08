import torch

from ..base import ConstantCurvature
from .utils import math_stereographic as math
from .frechetmean.frechet import FrechetMeanBall

from .utils.utils_ccs import EPS

class Stereographic(ConstantCurvature):
    """
    Represents the κ-stereographic model of a constant-curvature Riemannian manifold.
    This is borrowed from geoopt.

    This model unifies:
      - Euclidean space when κ = 0,
      - The Poincaré ball for hyperbolic geometry (κ < 0),
      - A stereographic projection of the hypersphere for spherical geometry (κ > 0).

    We use the convention ‖x‖² < - 1 / K, which means:
          - For κ < 0: ‖x‖² < - 1 / K,    (i.e., x lies inside the Poincaré ball)
          - For κ >= 0: R^n

    References
    ----------
    - A gyrovector space approach to hyperbolic geometry.
    - Hyperbolic neural networks.
    - Constant Curvature Graph Convolutional Networks
    - Mixed-curvature Variational Autoencoders
    - Differentiating through the Fréchet Mean
    - https://github.com/geoopt/geoopt
    - Riemannian Batch Normalization: A Gyro Approach
    """
    def __init__(self, K=-1.0, edge_eps=1e-6):
        super().__init__(K=K)
        self.min_norm = 1e-15
        self.edge_eps = edge_eps

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5, dim=-1):
        if self.K.lt(0):
            # for the Poincare ball
            norm_sq = x.pow(2).sum(dim=dim, keepdim=True)
            radius_sq = 1.0 / (-self.K)
            ok = (norm_sq < radius_sq + atol).all()
            reason = None if ok else f"‖x‖² = {norm_sq.max().item():.5f} ≥ 1 / |K| = {radius_sq:.5f}"
        else:
            ok, reason = True, None
        return ok, reason

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5, dim=-1):
        """
        For the Poincaré ball, every vector in R^n is a valid tangent vector.
        So we just check shape consistency.
        """
        ok = x.shape == u.shape
        reason = None if ok else f"x.shape = {x.shape} ≠ u.shape = {u.shape}"
        return ok, reason

    def random_tangent_origin(self, *size, mean=0, std=1, scale=1):
        tangent = torch.randn(*size, device=self.K.device, dtype=self.K.dtype) * std + mean
        return tangent*scale

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1]
        else:
            return sh

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1]
        else:
            return dim

    def zero(self, *shape):
        return torch.zeros(*shape)

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        return torch.zeros_like(x)

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def is_euclidean(self,):
        """Check if the current manifold is Euclidean."""
        return self.K.abs() < EPS[self.K.dtype]

    def lambda_x(self, x, keepdim=False):
        return math.lambda_x(x, k=self.K, dim=-1, keepdim=keepdim)

    def inner(self, x, u, v, keepdim=False):
        return math.inner(x, u, v, k=self.K, keepdim=keepdim, dim=-1)

    def inner0(self, u, v, keepdim=False):
        return 4 * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def proju0(self, u):
        return u

    def projx(self, x):
        if self.K.lt(0):
            #for poincare ball
            norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
            maxnorm = (1 - self.edge_eps) / (-self.K).sqrt()
            cond = norm > maxnorm
            projected = x / norm * maxnorm
            return torch.where(cond, projected, x)
        else:
            return x

    def egrad2rgrad(self, x, u):
        return math.egrad2rgrad(x, u, k=self.K,dim=-1)

    def exp(self, x, u, project=False):
        res = math.expmap(x, u, k=self.K, dim=-1)
        if project:
            return self.projx(res)
        else:
            return res

    def exp0(self, u, project=False):
        res = math.expmap0(u, k=self.K, dim=-1)
        if project:
            return self.projx(res)
        else:
            return res

    def log(self, x, y):
        return math.logmap(x, y, k=self.K, dim=-1)

    def log0(self, x):
        return math.logmap0(x, k=self.K, dim=-1)

    def dist(self, x, y, squared=False, keepdim=False):
        dist = math.dist(x, y, k=self.K, keepdim=keepdim, dim=-1)
        return dist.pow(2) if squared else dist

    def dist0(self, x, squared=False, keepdim=False):
        dist = math.dist0(x, k=self.K, dim=-1, keepdim=keepdim)
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        return math.parallel_transport(x, y, u, k=self.K, dim=-1)

    def transpfrom0(self, x, u):
        return math.parallel_transport0(x, u, k=self.K, dim=-1)

    def squeeze_tangent(self, x):
        return x

    def unsqueeze_tangent(self, x):
        return x

    def gyroadd(self, x, y):
        return math.mobius_add(x, y, k=self.K, dim=-1)

    def gyroscalarprod(self, x, r):
        return math.mobius_scalar_mul(r, x, k=self.K, dim=-1)

    def gyroinv(self, x):
        return -x

    def gyration(self, u, v, w):
        return math.gyration(u, v, w, k=self.K, dim=-1)

    def gyrocoadd(self,x,y):
        return math.mobius_coadd(x, y, k=self.K, dim=-1)

    def gyro_matvec(self, m, x):
        return math.mobius_matvec(m, x, k=self.K, dim=-1)

    def frechet_mean(self,x,max_iter=1000,w=None, batchdim=[0]):
        if self.K.lt(0):
            if w is None:
                w = torch.ones(x.shape[:-1]).to(x)
            return FrechetMeanBall.apply(x, w, self.K,max_iter)
        elif self.is_euclidean():
            return x.mean(dim=0)
        else:
            # currently only support [bs,n]
            return self.karcher_mean(x,max_iter=max_iter,w=w,batchdim=batchdim)

    def radius_to_stereo(self, x):
        return math.radius_to_stereo(x, self.K)

    def stereo_to_radius(self, y):
        return math.stereo_to_radius(y, self.K)


