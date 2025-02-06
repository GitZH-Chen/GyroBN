from typing import Tuple
import torch as th
import torch.nn as nn
from geoopt.tensor import ManifoldParameter

from Geometry.Grassmannian.utilities import gr_identity_batch
from Geometry.Grassmannian.GrGyro import GrassmannianGyro


saved_grads = {}

def hook_fn(name):
    def hook(grad):
        saved_grads[name] = grad.clone()
    return hook

class RBNGr(nn.Module):
    """
    Based on Alg. 1 "Differentiating through the fréchet mean", which is a variant of SPDNetBN and ManifoldNorm.
    """
    def __init__(self, shape: Tuple[int, ...], exp_mode='cayley',
                 momentum: float = 0.1,karcher_steps=1, batchdim: Tuple[int, ...] = [0], eps=1e-5):
        """
            shape : [c, n, p];
            exp_mode: cayley, expm;
        """
        super(__class__, self).__init__()
        self.shape, self.momentum, self.karcher_steps, self.exp_mode, self.batchdim, self.eps = shape, momentum, karcher_steps,exp_mode, batchdim, eps
        channel, n, p = shape
        self.subspace_dim = p
        self.Gyro = GrassmannianGyro(n, p, eps=self.eps,exp_mode=self.exp_mode)

        channels, n, p = self.shape
        bias_shape = shape if len(self.batchdim) == 1 else shape[1:]
        tmp_bias = gr_identity_batch(*bias_shape)
        self.bias = ManifoldParameter(tmp_bias, manifold=self.Gyro)

        # Initialize running_mean, running_var, and shift
        if len(self.batchdim)==1:
            self.register_buffer('running_mean', gr_identity_batch(*shape))
            self.register_buffer('running_var', th.ones(channel, 1, 1))
            self.shift = nn.Parameter(th.ones(channel, 1, 1))
        else:
            self.register_buffer('running_mean', gr_identity_batch(n, p))
            self.register_buffer('running_var', th.ones(1))
            self.shift = nn.Parameter(th.ones(1))

    def forward(self, X):
        if (self.training):
            # compute batch mean
            batch_mean = self.cal_geom_mean(X,karcher_steps=self.karcher_steps)
            batch_var = self.cal_geom_var(X,batch_mean)
            # scaling and shifting factor
            shifting_factor = self.shift / (batch_var + self.eps).sqrt()
            mean = batch_mean
            # update the running_mean, running_mean_test
            self.updating_running_statistics(batch_mean, batch_var)

        else:
            shifting_factor = self.shift / (self.running_var + self.eps).sqrt()
            mean = self.running_mean

        X_normalized = self.normalization(X, self.bias, mean, shifting_factor)

        return X_normalized

    def cal_geom_mean(self, X, karcher_steps=1, alpha=1):
        '''
        Function computing the Karcher mean for a batch of data by the Karcher flow
        Input X is a batch of Gramanniann matrices (b,c,n,p) to average
        Output is (c,n,p) Riemannian mean
        '''
        grass = X.detach()
        if len(self.batchdim)==1:
            batch_mean = grass[0,...]
        else:
            batch_mean = grass[0,0,...]
        for step in range(karcher_steps):
            x_log = self.Gyro.logmap(batch_mean, grass)
            batch_mean_tan = x_log.mean(dim=self.batchdim)

            #------  Check stopping criterion ------#
            # eps = 1e-3
            # norms = th.norm(batch_mean_tan, dim=(-2, -1), p='fro')
            # num_satisfied = th.sum(norms < eps).item()
            # print(f"{grass.shape} matrices at step {step}: norms {norms.mean():.2f}/{norms.std():.2f} "
            #       f"max {norms.max():.2f} min {norms.min():.2f}, with {num_satisfied}/{batch_mean.shape[0]} satisfied")
            # if th.all(norms < eps):
            #     print(f"Stopping criterion met at step {step}")
            #     break

            batch_mean = self.Gyro.expmap(batch_mean, alpha * batch_mean_tan)

        return batch_mean

    def cal_geom_var(self, X,batch_mean):
        """Frechet variance w.r.t. Frechet mean"""
        dists = self.Gyro.dist(X.detach(),batch_mean.detach())
        var = dists.square().mean(dim=self.batchdim)
        if len(self.batchdim)==1:
            return var.unsqueeze(-1).unsqueeze(-1)
        else:
            return var

    def updating_running_statistics(self, batch_mean, batch_var=None):
        """updating running statistics"""
        with th.no_grad():
            # updating mean
            self.running_mean.data = self.Gyro.geodesic(self.running_mean, batch_mean, self.momentum)
            # updating var running_var = running_var + t(batch_var - running_var)
            self.running_var.data = (1 - self.momentum) * self.running_var + batch_var * self.momentum
    def normalization(self, U, B, M, shifting_factor):
        '''Exp_{B} {s PT_{M \rightarrow B} [Log _M (U)] }'''

        delta = self.Gyro.logmap_standard(M,U)

        Q,s_tilde, R = self.logmap_aux(M,B)
        cosSigma = th.cos(s_tilde).unsqueeze(-2)
        sinSigma = th.sin(s_tilde).unsqueeze(-2)
        
        lfs_ptrans = -(M @ R).mul(sinSigma) @ Q.transpose(-1,-2) + Q.mul(cosSigma) @ Q.transpose(-1,-2) + self.Gyro.In-Q @ Q.transpose(-1,-2)
        vec_pt =  lfs_ptrans @ delta
        U_new = self.Gyro.expmap(B, shifting_factor * vec_pt)
        return U_new

    def logmap_aux(self, x, y):
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

        return u,s_atan, vh.transpose(-1,-2)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, exp_mode={self.exp_mode}," \
               f"momentum={self.momentum}, karcher_steps={self.karcher_steps}, batchdim={self.batchdim},eps={self.eps:.1e})"


