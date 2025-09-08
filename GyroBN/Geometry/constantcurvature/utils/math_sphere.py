import torch
from torch import Tensor
import torch.jit

from .utils_ccs import EPS, sindiv

# =======================
# Low-level TorchScript ops
# =======================

@torch.jit.script
def _inner(u: Tensor, v: Tensor, keepdim: bool = False) -> Tensor:
    return (u * v).sum(dim=-1, keepdim=keepdim)

@torch.jit.script
def _norm(u: Tensor, keepdim: bool = False) -> Tensor:
    return u.norm(p=2, dim=-1, keepdim=keepdim)

@torch.jit.script
def _proju(x: Tensor, u: Tensor, K: Tensor) -> Tensor:
    inner = (u * x).sum(dim=-1, keepdim=True)
    return u - K * inner * x

@torch.jit.script
def _proju0(u: Tensor) -> Tensor:
    zero_time = torch.zeros_like(u[..., :1])
    spatial = u[..., 1:]
    return torch.cat([zero_time, spatial], dim=-1)

@torch.jit.script
def _projx(x: Tensor, K: Tensor, eps: float) -> Tensor:
    R = 1.0 / K.sqrt()
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm * R

@torch.jit.script
def _exp(x: Tensor, u: Tensor, K: Tensor, eps: float) -> Tensor:
    u_norm = u.norm(p=2, dim=-1, keepdim=True)
    alpha = (u_norm * K.sqrt()).clamp_min(eps)
    return torch.cos(alpha) * x + sindiv(alpha) * u

@torch.jit.script
def _exp0(v: Tensor, K: Tensor, eps: float) -> Tensor:
    vs = v[..., 1:]  # space component
    vs_norm = vs.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    theta = vs_norm * K.sqrt()
    time = torch.cos(theta) / (K.sqrt())
    space = sindiv(theta) * vs
    return torch.cat([time, space], dim=-1)

@torch.jit.script
def _log(x: Tensor, y: Tensor, K: Tensor, eps: float) -> Tensor:
    beta = ((x * y).sum(dim=-1, keepdim=True) * K).clamp(-1.0+eps, 1.0-eps)  # β = K⟨x, y⟩
    sin = (1.0 - beta**2).sqrt().clamp_min(eps)
    coef = torch.acos(beta) / sin
    return coef * (y - beta * x)

@torch.jit.script
def _log0(x: Tensor, K: Tensor, eps: float) -> Tensor:
    xt = x[..., :1]  # time component
    xs = x[..., 1:]  # space component
    alpha = (xt * K.sqrt()).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(alpha)
    xs_norm = xs.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    spatial = theta / (xs_norm * K.sqrt()) * xs
    time = torch.zeros_like(xt)
    return torch.cat([time, spatial], dim=-1)

@torch.jit.script
def _dist(x: Tensor, y: Tensor, K: Tensor, eps: float, keepdim: bool = False) -> Tensor:
    inner = (x * y).sum(dim=-1, keepdim=keepdim).clamp(-1.0 + eps, 1.0 - eps)
    return (1.0 / K.sqrt()) * torch.acos(inner * K)

@torch.jit.script
def _dist0(x: Tensor, K: Tensor, eps: float, keepdim: bool = False) -> Tensor:
    xt = x[..., :1] if keepdim else x[..., 0]
    alpha = (xt * K.sqrt()).clamp(-1.0 + eps, 1.0 - eps)
    return (1.0 / K.sqrt()) * torch.acos(alpha)

@torch.jit.script
def _transp(x: Tensor, y: Tensor, v: Tensor, K: Tensor, eps: float) -> Tensor:
    xy_inner = (x * y).sum(dim=-1, keepdim=True)       # ⟨x, y⟩
    yv_inner = (y * v).sum(dim=-1, keepdim=True)       # ⟨y, v⟩
    denom = (1 + K * xy_inner).clamp_min(eps)
    correction = (K * yv_inner / denom) * (x + y)
    return v - correction

@torch.jit.script
def _transpfrom0(x: Tensor, v: Tensor, K: Tensor, eps: float) -> Tensor:
    xt = x[..., :1]  # time component
    xs = x[..., 1:]  # space component
    vs = v[..., 1:]  # spatial part of tangent vector
    R = (1.0 / K).sqrt()
    dot = (xs * vs).sum(dim=-1, keepdim=True)
    denom = (1 + xt * K.sqrt()).clamp_min(eps)
    coef = (K * dot) / denom
    time = coef * (xt + R)
    space = coef * xs
    return v - torch.cat([time, space], dim=-1)

@torch.jit.script
def _gyroscalarprod(x: Tensor, r: Tensor, K: Tensor, eps: float) -> Tensor:
    xt = x[..., :1]         # time component
    xs = x[..., 1:]         # space component

    sqrtK = K.sqrt()
    alpha = (xt * sqrtK).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(alpha)              # θ = arccos_K(√K x_t)
    rtheta = r * theta                     # scaled angle

    xs_norm = xs.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

    time = (1.0 / sqrtK) * torch.cos(rtheta)
    space = (1.0 / sqrtK) * torch.sin(rtheta) * xs / xs_norm
    return torch.cat([time, space], dim=-1)

@torch.jit.script
def _gyroinv(x: Tensor) -> Tensor:
    xt = x[..., :1]
    xs = x[..., 1:]
    return torch.cat([xt, -xs], dim=-1)

# =======================
# Public base-compatible wrappers
# =======================

def inner(u: Tensor, v: Tensor = None, keepdim: bool = False) -> Tensor:
    return _inner(u, v, keepdim=keepdim)

def norm(u: Tensor, keepdim: bool = False) -> Tensor:
    return _norm(u, keepdim=keepdim)

def proju(x: Tensor, u: Tensor, K: Tensor) -> Tensor:
    return _proju(x, u, K)

def proju0(u: Tensor) -> Tensor:
    return _proju0(u)

def projx(x: Tensor, K: Tensor) -> Tensor:
    return _projx(x, K, EPS[x.dtype])

def exp(x: Tensor, u: Tensor, K: Tensor) -> Tensor:
    return _exp(x, u, K, EPS[x.dtype])

def exp0(u: Tensor, K: Tensor) -> Tensor:
    return _exp0(u, K, EPS[u.dtype])

def log(x: Tensor, y: Tensor, K: Tensor) -> Tensor:
    return _log(x, y, K, EPS[x.dtype])

def log0(x: Tensor, K: Tensor) -> Tensor:
    return _log0(x, K, EPS[x.dtype])

def dist(x: Tensor, y: Tensor, K: Tensor, keepdim: bool = False) -> Tensor:
    return _dist(x, y, K, EPS[x.dtype], keepdim=keepdim)

def dist0(x: Tensor, K: Tensor, keepdim: bool = False) -> Tensor:
    return _dist0(x, K, EPS[x.dtype], keepdim=keepdim)

def transp(x: Tensor, y: Tensor, v: Tensor, K: Tensor) -> Tensor:
    return _transp(x, y, v, K, EPS[x.dtype])

def transpfrom0(x: Tensor, v: Tensor, K: Tensor) -> Tensor:
    return _transpfrom0(x, v, K, EPS[x.dtype])

def gyroscalarprod(x: Tensor, r: Tensor, K: Tensor) -> Tensor:
    return _gyroscalarprod(x, r, K, EPS[x.dtype])

def gyroinv(x: Tensor) -> Tensor:
    return _gyroinv(x)

