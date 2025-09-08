import torch

from ..base import ConstantCurvature
from .utils import math_sphere as math

from .utils.math_stereographic import mobius_add, radius_to_stereo, radius_to_stereo
from .utils.math_hyperboloid import gyroadd_radius

class Sphere(ConstantCurvature):
    """
    Represents the Sphere model with constant positive curvature.
    The sphere is defined by the constraint ‖x‖² = 1 / K, where K >0 is the curvature and R = 1/√K is the radius.

    Parameters
    ----------
    K : float, default=1.0
        Curvature of the sphere (must be positive)
    edge_eps : float, default=1e-6
        Small epsilon value for numerical stability in boundary operations

    References
    ----------
    - Mixed-curvature Variational Autoencoders
    - https://github.com/oskopek/mvae
    - Riemannian Batch Normalization: A Gyro Approach
    """
    def __init__(self, K=1.0, edge_eps=1e-6):
        super().__init__(K=K)
        assert K > 0
        self.min_norm = 1e-15
        self.edge_eps = edge_eps

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5, dim=-1):
        norm_sq = x.pow(2).sum(dim=dim, keepdim=True)
        radius_sq = 1.0 / self.K

        ok = torch.allclose(norm_sq, radius_sq, atol=atol, rtol=rtol)
        reason = None if ok else f"‖x‖² = {norm_sq.max().item():.5f} ≠ 1 / K = {radius_sq.item():.5f}"
        return ok, reason

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5, dim=-1):
        inner_prod = math.inner(x, u, keepdim=True)  # 这里是⟨x, u⟩
        ok = torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=atol, rtol=rtol)
        reason = None if ok else f"|⟨x, u⟩| = {inner_prod.abs().max().item():.5e} ≠ 0"
        return ok, reason

    def random_tangent_origin(self, *size, mean=0, std=1, scale=1):
        n = size[-1] # intrinsic dimension
        # Sample spatial part in R^n
        spatial = torch.randn(*size[:-1], n, device=self.K.device, dtype=self.K.dtype) * std + mean
        # Pad zero in the time coordinate
        tangent = torch.cat([
            torch.zeros(*size[:-1], 1, device=self.K.device, dtype=self.K.dtype),  # time part = 0
            spatial
        ], dim=-1)
        return tangent * scale

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1] - 1
        else:
            return sh - 1

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1] + 1
        else:
            return dim + 1

    def zero(self, *shape):
        x = torch.zeros(*shape, dtype=self.K.dtype, device=self.K.device)
        x[..., 0] = 1 / self.K.sqrt()
        return x

    def zero_tan(self, *shape):
        return torch.zeros(*shape, dtype=self.K.dtype, device=self.K.device)

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = 1 / self.K.sqrt()
        return y

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def inner(self, x, u, v, keepdim=False):
        return math.inner(u, v, keepdim=keepdim)

    def inner0(self, u, v, keepdim=False):
        return math.inner(u, v, keepdim=keepdim)

    def proju(self, x, u):
        return math.proju(x, u, self.K)

    def proju0(self, u):
        """" this is K-invariant"""
        return math.proju0(u)

    def projx(self, x):
        return math.projx(x, self.K)

    egrad2rgrad = proju

    def exp(self, x, u, project=False):
        res = math.exp(x, u, self.K)
        if project:
            return self.projx(res)
        else:
            return res

    def exp0(self, u, project=False):
        res = math.exp0(u, self.K)
        if project:
            return self.projx(res)
        else:
            return res

    def log(self, x, y):
        return math.log(x, y,self.K)

    def log0(self, x):
        return math.log0(x, self.K)

    def dist(self, x, y, squared=False, keepdim=False):
        dist = math.dist(x, y, self.K, keepdim=keepdim)
        return dist.pow(2) if squared else dist

    def dist0(self, x, squared=False, keepdim=False):
        dist = math.dist0(x, self.K, keepdim=keepdim)
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        return math.transp(x, y, u, self.K)

    def transpfrom0(self, x, u):
        return math.transpfrom0(x, u, self.K)

    def gyroscalarprod(self, x, r):
        return math.gyroscalarprod(x, r, self.K)
    def gyroinv(self, x):
        return math.gyroinv(x)

    def _gyroadd_via_iso(self, x, y):
        """Calculating gyroadd via isomorphism"""
        x_p = radius_to_stereo(x, self.K)
        y_p = radius_to_stereo(y, self.K)
        res_p = mobius_add(x_p, y_p, k=self.K, dim=-1)
        return radius_to_stereo(res_p, self.K)

    def _gyroadd_via_riem(self, x, y):
        """Calculating gyroadd via isomorphism"""
        u = self.log0(y)
        v = self.transpfrom0(x, u)
        return self.exp(x, v)

    def gyroadd(self, x, y):
        return gyroadd_radius(x, y, self.K)

    def gyro_matvec(self, W, x):
        u = self.log0(x)  # [B, n+1]
        spatial = u[..., 1:]  # [B, n]
        zero_time = torch.zeros_like(spatial[..., :1])  # [B, 1]
        Wx = torch.matmul(spatial, W.T)  # Matrix multiplication
        lifted = torch.cat([zero_time, Wx], dim=-1)  # [B, m+1]
        return self.exp0(lifted)


