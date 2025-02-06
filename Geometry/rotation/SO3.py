"""
    Official Implementation of the rotation matrices used in
    @inproceedings{chen2024rmlr,
        title={{RMLR}: Extending Multinomial Logistic Regression into General Geometries},
        author={Ziheng Chen and Yue Song and Rui Wang and Xiaojun Wu and Nicu Sebe},
        booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
        year={2024},
    }
"""


import torch as th
from pytorch3d.transforms import matrix_to_axis_angle
from geoopt.manifolds import Manifold
from typing import Optional, Tuple, Union
saved_grads = {}

def hook_fn(name):
    def hook(grad):
        saved_grads[name] = grad.clone()
    return hook


class Rotation(Manifold):
    """
    Computation for SO3 data with size of [...,n,n]
    Following manopt, we use the Lie algebra representation for the tangent spaces
    """
    # __scaling__ = Manifold.__scaling__.copy()
    name = "Rotation"
    ndim = 2
    reversible = False
    def __init__(self,eps=1e-8):
        super().__init__()
        self.eps=eps;
        self.register_buffer('running_mean', th.eye(3,3))

    def rand_rotations(self,num):
        """generating 3D random rotations"""
        A = th.rand(num, 3, 3)
        u, _, v = th.linalg.svd(A.matmul(A.transpose(-1, -2)))
        u_det = u.det()
        # indx = self.is_not_equal(u_det,1.)
        # u[indx] = -u[indx]
        # u_new = th.einsum('ijk,i->ijk', u, th.sign(u_det))
        u_new = th.einsum('i...,i->i...', u, th.sign(u_det))
        result,_ = self._check_point_on_manifold(u_new)
        if not result:
            raise ValueError("SO3  init value error")
        # U = random_rotations(n)
        # A=th.randn(b,n,f,3,3)
        # U, S, V = th.linalg.svd(A.matmul(A.transpose(-1,-2))+1e-6*th.eye(3))
        return u_new
    def rand_skrew_sym(self,n,f):
        A=th.rand(n,f,3,3)
        return A-A.transpose(-1,-2)

    def quat_axis2log_axis(self,axis_quat):
        """"
        convert quaternion axis into matrix log axis
        https://github.com/facebookresearch/pytorch3d/issues/188
        """
        log_axis_result = axis_quat.clone()
        angle_in_2pi = log_axis_result.norm(p=2, dim=-1, keepdim=True)
        mask = angle_in_2pi > th.pi
        tmp = 1 - 2 * th.pi / angle_in_2pi
        new_values = tmp * log_axis_result
        results = th.where(mask, new_values, log_axis_result)

        # log_axis_result[mask] = th.einsum('...km,...k->km', log_axis_result[mask], 1 - 2 * th.pi / angle_in_2pi[mask])
        # tmp_result = is_equal(axis_quat, log_axis_result)
        return results
    def matrix2euler_axis(self,R):
        quat_vec = matrix_to_axis_angle(R)
        log_axis = self.quat_axis2log_axis(quat_vec)
        euler_axis = log_axis.div(log_axis.norm(dim=-1,keepdim=True))
        return euler_axis

    def mLog(self, R):
        """
        Note that Exp(\alpha A)=Exp(A), with \alpha = \frac{\theta-2\pi}{\theta}
        So, for a single rotation matrices, quat_axis or log_axis does not affect self.mExp.
        """
        vec = matrix_to_axis_angle(R)
        log_vec = self.quat_axis2log_axis(vec)
        skew_symmetric = self.vec2skrew(log_vec)
        return skew_symmetric

    def vec2skrew(self,vec):
        # skew_symmetric = th.zeros_like(vec).unsqueeze(-1).expand(*vec.shape, 3).contiguous()
        skew_symmetric = th.zeros(*vec.shape, 3,dtype=vec.dtype,device=vec.device)
        skew_symmetric[..., 0, 1] = -vec[..., 2]
        skew_symmetric[..., 1, 0] = vec[..., 2]
        skew_symmetric[..., 0, 2] = vec[..., 1]
        skew_symmetric[..., 2, 0] = -vec[..., 1]
        skew_symmetric[..., 1, 2] = -vec[..., 0]
        skew_symmetric[..., 2, 1] = vec[..., 0]
        return skew_symmetric
    def mExp(self, S):
        """Computing matrix exponential for skrew symmetric matrices"""
        a, b, c = S[..., 0, 1], S[..., 0, 2], S[..., 1, 2]
        theta = th.sqrt(a ** 2 + b ** 2 + c ** 2).unsqueeze(-1).unsqueeze(-1)

        S_normalized = S / theta
        S_norm_squared = S_normalized.matmul(S_normalized)
        sin_theta = th.sin(theta)
        cos_theta = th.cos(theta)
        tmp_S = self.I + sin_theta * S_normalized + (1 - cos_theta) * S_norm_squared

        S_new = th.where(theta < self.eps, S-S.detach()+self.I, tmp_S) # S+I to ensure autograd

        return S_new

    def transp(self, x, y, v):
        return v

    def inner(self, x, u, v, keepdim=False):
        if v is None:
            v = u
        return th.sum(u * v, dim=[-2, -1], keepdim=keepdim)

    def projx(self, x):
        u, s, vt = th.linalg.svd(x)
        ones = th.ones_like(s)[..., :-1]
        signs = th.sign(th.det(th.matmul(u, vt))).unsqueeze(-1)
        flip = th.cat([ones, signs], dim=-1)
        result  = u.matmul(th.diag_embed(flip)).matmul(vt)
        return result

    def proju(self,X, H):
        k = self.multiskew(H)
        return k

    def egrad2rgrad(self, x, u):
        """Map the Euclidean gradient :math:`u` in the ambient space on the tangent
        space at :math:`x`.
        """
        k = self.multiskew(x.transpose(-1, -2).matmul(u))
        return k

    def retr(self, X,U):
        Y = X + X.matmul(U)
        Q, R = th.linalg.qr(Y)
        New = th.matmul(Q, th.diag_embed(th.sign(th.sign(th.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return New

    def multiskew(self,A):
        return 0.5 * (A - A.transpose(-1,-2))

    def logmap(self,R,S):
        """ return skrew symmetric matrices """
        return self.mLog(R.transpose(-1,-2).matmul(S))

    def expmap(self,R,V):
        """ V is the skrew symmetric matrices """
        return R.matmul(self.mExp(V))

    def geodesic(self,R,S,t):
        """ the geodesic connecting R and s """
        vector = self.logmap(R,S)
        X_new = R.matmul(self.mExp(t*vector))
        return X_new

    def trace(self,m):
        """Computation for trace of m of [...,n,n]"""
        return th.einsum("...ii->...", m)

    def cal_roc_angel_batch(self,r,epsilon=1e-4):
        """
        Following the matlab implemetation, we set derivative=0 for tr near -1 or near 3.
        Besides, there could be cases where tr \in [-1-eps, 3+eps] due to numerical error.
        We view the cases beyond the [-1,3] as -1 or 3, and the derivative is 0.
        return:
            tr <= -1-epsilon, theta=pi (derivative is 0)
            tr >= 3-epsilon, theta=0 (derivative is 0)
        """
        assert epsilon >= 0, "Epsilon must be positive"

        mtrc = self.trace(r)

        maskpi = (mtrc + 1) <= epsilon  # tr <= -1 + epsilon
        # mtrcpi = -mtrc * maskpi * np.pi # this is different from the matlab implemetation, as its direvative is -pi
        mtrcpi = maskpi * th.pi
        maskacos = ((mtrc + 1) > epsilon) * ((3- mtrc) > epsilon) # -1+epsilon < tr < 3 - epsilon

        mtrcacos = th.acos((mtrc * maskacos - 1) / 2) * maskacos # -1+epsilon < tr < 3 - epsilon, use the acos
        results = mtrcpi + mtrcacos # for tr  -tr <= -1 + epsilon and tr >= 3-epsilon, the derivative = 0
        return results

    def _check_point_on_manifold(
            self, x: th.Tensor, *, atol=1e-5, rtol=1e-8
        ) -> Union[Tuple[bool, Optional[str]], bool]:

        if x.shape[-1] != 3 or x.shape[-2] != 3:
            raise ValueError("Input matrices must be 3x3.")

        # Check orthogonality
        is_orthogonal = th.allclose(x @ x.transpose(-1, -2),
                                       th.eye(3, device=x.device),
                                       atol=atol,rtol=rtol)

        # Check determinant
        det = th.det(x)
        is_det_one = th.allclose(det, th.tensor(1.0, device=x.device), atol=atol,rtol=rtol)

        # Combine both conditions
        is_SO3 = is_orthogonal & is_det_one

        return is_SO3, None

    def _check_vector_on_tangent(
            self, x: th.Tensor, u: th.Tensor, *, atol=1e-5, rtol=1e-8
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """Check whether u is a skrew symmetric matrices"""
        diff = u + u.transpose(-1,-2)
        ok = th.allclose(diff, th.zeros_like(diff), atol=atol, rtol=rtol)
        return ok, None

    def is_not_equal(self, a, b, eps=0.01):
        """ Return true if not eaqual"""
        return th.nonzero(th.abs(a - b) > eps)


