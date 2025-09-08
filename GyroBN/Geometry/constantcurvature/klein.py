"""Klein Model."""

import torch

from ..base import Hyperbolic
from .utils.utils_ccs import EPS,artanh, tanh

from .frechetmean.frechet import FrechetMeanBall


class Klein(Hyperbolic):
    """
    Klein model class.  x0^2 + x1^2 + ... + xd^2 < -1 / K
    Several operations are identical with the Poincare ball, including
        gyroinv, gyroscalarprod
        log0, exp0

    References
    ----------
    - Introduction to Riemannian manifolds
    - Ziheng Chen, et al., Riemannian Batch Normalization: A Gyro Approach 2025
    """

    def __init__(self, K=-1.0):
        super(__class__, self).__init__(K=K)
        self.edge_eps = 1e-6
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5, dim=-1):
        """
        Check whether a point x lies inside the Poincaré ball: ‖x‖² < 1 / |K|
        """
        norm_sq = x.pow(2).sum(dim=dim, keepdim=True)
        radius_sq = 1.0 / (-self.K)
        ok = (norm_sq < radius_sq + atol).all()
        reason = None if ok else f"‖x‖² = {norm_sq.max().item():.5f} ≥ 1 / |K| = {radius_sq:.5f}"
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

    def proju(self, x, u):
        return u

    def proju0(self, u):
        return u

    def projx(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps) / (-self.K).sqrt()
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    # === Below is for Isometries===
    def klein_to_poincare(self, x):
        norm_x = x.norm(dim=-1, keepdim=True, p=2)  # Compute the norm of x
        return x / (torch.sqrt(1 + self.K * norm_x ** 2) + 1)

    def poincare_to_klein(self, x):
        norm_x = x.norm(dim=-1, keepdim=True, p=2)  # Compute the norm of x
        return (2 * x) / (1 - self.K * norm_x ** 2)

    def d_klein_to_poincare(self, x, u):
        """Differential of the map π_{K→P} at point x acting on tangent vector u."""
        K = self.K
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
        sqrt_term = torch.sqrt(1 + K * x_norm_sq)
        denom1 = (1 + sqrt_term)
        denom2 = (1 + sqrt_term).pow(2) * sqrt_term
        x_dot_u = (x * u).sum(dim=-1, keepdim=True)

        term1 = u / denom1
        term2 = (K * x_dot_u / denom2) * x
        return term1 - term2

    def d_poincare_to_klein(self, x, u):
        """Differential of the map π_{P→K} at point x acting on tangent vector u."""
        K = self.K
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
        denom1 = (1 - K * x_norm_sq)
        denom2 = denom1.pow(2)
        x_dot_u = (x * u).sum(dim=-1, keepdim=True)

        term1 = (2 / denom1) * u
        term2 = (4 * K * x_dot_u / denom2) * x
        return term1 + term2

    # === Below is for Riemannian opeartors===

    def gamma_x(self, x, dim=-1, keepdim=False):
        return 1 / torch.sqrt(1 + self.K * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def inner(self, x, u, v, keepdim=False):
        K = self.K
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)  # ||x||²
        denom1 = 1 + K * x_norm_sq
        denom2 = denom1.pow(2)

        dot_uv = (u * v).sum(dim=-1, keepdim=True)
        dot_xu = (x * u).sum(dim=-1, keepdim=True)
        dot_xv = (x * v).sum(dim=-1, keepdim=True)

        result = dot_uv / denom1 - K * dot_xu * dot_xv / denom2
        if not keepdim:
            result = result.squeeze(-1)
        return result

    def inner0(self, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def exp(self, x, u, project=False):
        K = self.K
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)  # ||x||^2
        sqrt_term = torch.sqrt(1 + K * x_norm_sq)  # √(1 + K||x||²)
        coef1 = 1 / sqrt_term
        dot = (x * u).sum(dim=-1, keepdim=True)
        coef2 = K * dot / ((1 + sqrt_term) * (1 + K * x_norm_sq))
        u_tilde = coef1 * u - coef2 * x

        exp0 = self.exp0(u_tilde)
        out = self.gyroadd(x, exp0)  # Use Poincare Mobius addition

        if project:
            out = self.projx(out)
        return out

    def exp0(self, u, project=False):
        sqrtK = (-self.K) ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min=EPS[u.dtype])
        theta = sqrtK * u_norm
        gamma_1 = (tanh(theta) / theta) * u
        if project:
            return self.projx(gamma_1)
        else:
            return gamma_1

    def log(self, x, y):
        log0 = self.log0(self.gyroadd(-x, y))
        x_poincare = self.klein_to_poincare(x)
        lambda_x_poincare = self.lambda_x(x_poincare, keepdim=True)
        transported = self.d_poincare_to_klein(x_poincare, log0)
        return transported / lambda_x_poincare

    def log0(self, x):
        sqrtK = (-self.K) ** 0.5
        x_norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(EPS[x.dtype])
        y = sqrtK * x_norm
        scale = artanh(y) / y
        return scale * x
    def transp(self, x, y, u):
        p_x = self.klein_to_poincare(x)
        p_y = self.klein_to_poincare(y)
        tan_poincare = self.poincare_transp(p_x,p_y,self.d_klein_to_poincare(x,u))
        return self.d_poincare_to_klein(p_y,tan_poincare)

    def dist(self, x, y, squared=False, keepdim=False):
        sqrt_c = (-self.K) ** 0.5
        add_norm = self.gyroadd(-x, y).norm(dim=-1, p=2, keepdim=keepdim)
        dom = 1 + torch.sqrt(1 + self.K * add_norm **2)
        dist_c = artanh(sqrt_c * add_norm / dom)
        dist = dist_c * 2 / sqrt_c
        return dist.pow(2) if squared else dist

    def dist0(self, x, squared=False, keepdim=False):
        sqrt_c = (-self.K) ** 0.5
        add_norm = x.norm(dim=-1, p=2, keepdim=keepdim)
        dom = 1 + torch.sqrt(1 + self.K * add_norm **2)
        dist_c = artanh(sqrt_c * add_norm / dom)
        dist = dist_c * 2 / sqrt_c
        return dist.pow(2) if squared else dist

    # === Below is for Gyro opeartors===
    def gyroadd(self, x, y, dim=-1):
        """Einstein add"""
        xy = (x * y).sum(dim=dim, keepdim=True) # <x, y>
        gamma_x = self.gamma_x(x,dim=dim,keepdim=True) # 1 / torch.sqrt(1 + self.K * x2)
        num = x + (1 / gamma_x) * y - self.K * (gamma_x / (1 + gamma_x)) * xy * x
        denom = 1 - self.K * xy
        return num / denom.clamp_min(self.min_norm)

    def gyroscalarprod(self, x, r):
        """Einstein scalar multiplication"""
        sqrt_minus_K = torch.sqrt(-self.K)
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        tanh_part = torch.tanh(r * torch.atanh(sqrt_minus_K * norm_x))
        scaled_x = (tanh_part / (sqrt_minus_K * norm_x)) * x
        return scaled_x

    def gyroinv(self, x):
        return -x

    def gyration(self, u, v, w):
        K = self.K
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        gamma_u = self.gamma_x(u,keepdim=True)
        gamma_v = self.gamma_x(v,keepdim=True)

        A = (
                K * (gamma_u ** 2 / (gamma_u + 1)) * (gamma_v - 1) * uw
                - K * gamma_u * gamma_v * vw
                + 2 * K ** 2 * (gamma_u ** 2 * gamma_v ** 2 / ((gamma_u + 1) * (gamma_v + 1))) * uv * vw
        )
        B = (
                K * (gamma_v / (gamma_v + 1)) *
                (gamma_u * (gamma_v + 1) * uw + (gamma_u - 1) * gamma_v * vw)
        )
        D = 1 + gamma_u * gamma_v * (1 - K * uv )

        return w + (A * u + B * v) / D

    # === Others ===

    def frechet_mean(self,x,max_iter=1000,w=None):
        x_poincare = self.klein_to_poincare(x)
        if w is None:
            w = torch.ones(x.shape[:-1]).to(x)
        mu_poincare = FrechetMeanBall.apply(x_poincare, w, self.K,max_iter)
        return self.poincare_to_klein(mu_poincare)

    def mobius_add(self, x, y):
        '''mobius addition on the Poincaré ball'''
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * self.K * xy - self.K * y2) * x + (1 + self.K * x2) * y
        denom = 1 - 2 * self.K * xy + (self.K.pow(2)) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def lambda_x(self, x, keepdim=False):
        return 2 / (1 + self.K * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def gyro_matvec(self, m, x):
        """Einstein_matvec share the same expression as the mobius_matvec"""
        sqrt_c = (-self.K) ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def poincare_transp(self, x, y, u):
        return (
            self.poincare_gyration(y, -x, u)
            * self.lambda_x(x, keepdim=True)
            / self.lambda_x(y, keepdim=True)
        )

    def poincare_gyration(self, u, v, w):
        u2 = u.pow(2).sum(dim=-1, keepdim=True)
        v2 = v.pow(2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = - self.K.pow(2) * uw * v2 - self.K * vw + 2 * self.K.pow(2) * uv * vw
        b = - self.K.pow(2) * vw * u2 + self.K * uw
        d = 1 - 2 * self.K * uv + self.K.pow(2) * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])





