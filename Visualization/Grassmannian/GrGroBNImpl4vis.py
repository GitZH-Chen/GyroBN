from typing import Tuple
import torch as th
import torch.nn as nn

from Geometry.Grassmannian.utilities import gr_identity_batch
from Geometry.Grassmannian.GrGyro import GrassmannianGyro

saved_grads = {}

def hook_fn(name):
    def hook(grad):
        saved_grads[name] = grad.clone()
    return hook

class GrGyroBNImpl(nn.Module):
    """
    Implementation of GyroBN on the Grassmannian, input are expected to be [b,c,n,p]
    KF and gyro operations are caculated by ONB perspective
    Biasing parameter are parameterized by the exponential coordinate at \tilde{I}_{n,p}
    """
    def __init__(self, shape: Tuple[int, ...], momentum: float = 0.1,karcher_steps=1,
                 exp_mode='cayley', batchdim: Tuple[int, ...] = [0], eps=1e-5):
        """
            shape : [c, n, p];
            exp_mode: cayley, expm;
            param_mode='trivial', 'Grassmannian'
            init_mode='identity', 'uniform', 'normal'
        """
        super(__class__, self).__init__()
        self.shape, self.momentum, self.karcher_steps, self.exp_mode, self.batchdim, self.eps = shape, momentum, karcher_steps,exp_mode, batchdim, eps
        n, p = shape
        self.Gyro = GrassmannianGyro(n, p, eps=self.eps,exp_mode=self.exp_mode)

        # Initialize bias and shift as parameters,
        # self.bias = geoopt.ManifoldParameter(functionals.gr_identity_batch(shape),manifold=Grassmannian(n, p, eps=self.eps))
        # Trivilization of bias \in \bbR^{n-p,p}, and 0 corresponds to the initial value as identity element
        n, p = self.shape
        bias_shape=[n-p,p] if len(self.batchdim)==1 else [n-p,p]
        tmp_bias = th.zeros(bias_shape)
        self.bias = nn.Parameter(tmp_bias)

        # Initialize running_mean, running_var, and shift
        if len(self.batchdim)==1:
            self.register_buffer('running_mean', gr_identity_batch(*shape))
            self.register_buffer('running_var', th.ones(1, 1))
            self.shift = nn.Parameter(th.tensor(0.2))
        else:
            self.register_buffer('running_mean', gr_identity_batch(n, p))
            self.register_buffer('running_var', th.ones(1))
            self.shift = nn.Parameter(th.ones(1))

    def forward(self, X):
        if (self.training):
            # compute batch mean
            batch_mean = self.cal_geom_mean(X,karcher_steps=self.karcher_steps)
            batch_var = self.cal_geom_var(X,batch_mean)
            # centering
            X_centered = self.Gyro.left_gyrotranslation_V2U(batch_mean,X, is_inverse=True)
            # scaling and shifting
            factor = self.shift / (batch_var+self.eps).sqrt()
            X_scaled = self.Gyro.gyro_scalarproduct(factor, X_centered)
            # update the running_mean, running_mean_test
            self.updating_running_statistics(batch_mean, batch_var)
        else:
            # centering, scaling and shifting
            X_centered = self.Gyro.left_gyrotranslation_V2U(self.running_mean, X, is_inverse=True)
            factor = self.shift / (self.running_var + self.eps).sqrt()
            X_scaled = self.Gyro.gyro_scalarproduct(factor, X_centered)

        skew_bias = self.Gyro.B2skrew(self.bias)
        X_normalized = self.Gyro.left_gyrotranslation_skew2U(skew_bias, X_scaled, is_inverse=False)

        return X_normalized,batch_mean

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
            batch_mean = grass[0,...]
        for step in range(karcher_steps):
            x_log = self.Gyro.logmap(batch_mean, grass)
            batch_mean_tan = x_log.mean(dim=self.batchdim)
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


