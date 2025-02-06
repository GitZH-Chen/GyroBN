import torch
import torch.nn as nn
import math

from frechetmean.manifolds import Poincare, Lorentz
from frechetmean.frechet import frechet_mean


class GyroBNH(nn.Module):
    def __init__(self, dim, manifold,momentum=0.1,eps=1e-6):
        super(__class__, self).__init__()
        self.man = manifold;self.dim=dim;self.momentum=momentum;self.eps=eps

        self.mean = nn.Parameter(self.man.zero_tan(self.man.dim_to_sh(dim)))
        self.var = nn.Parameter(torch.tensor(0.5))

        # statistics
        self.running_mean = None
        self.running_var = None
        self.updates = 0

    def forward(self, x):

        if self.training:
            # frechet mean, use iterative and don't batch (only need to compute one mean)
            input_mean = frechet_mean(x, self.man)
            input_var = self.man.frechet_variance(x, input_mean)
            self.updating_running_statistics(input_mean,input_var)
        else:
            if self.updates == 0:
                raise ValueError("must run training at least once")
            input_mean = self.running_mean
            input_var = self.running_var

        output = self.normalization(x,input_mean,input_var)
        return output,input_mean

    def normalization(self,x,input_mean,input_var):
        on_manifold = self.man.exp0(self.mean)
        # factor = (self.var / (input_var + self.eps)).sqrt()
        factor = self.var / (input_var + self.eps).sqrt()
        x_center = self.man.mobius_addition(-input_mean,x)
        x_scaled = self.man.mobius_scalar_mul(factor, x_center)
        x_normed = self.man.mobius_addition(on_manifold, x_scaled)
        return x_normed

    def updating_running_statistics(self, batch_mean, batch_var):
        self.updates += 1
        if self.running_mean is None:
            self.running_mean = batch_mean
        else:
            self.running_mean = self.man.exp(
                self.running_mean,
                self.momentum * self.man.log(self.running_mean, batch_mean)
            )
        if self.running_var is None:
            self.running_var = batch_var
        else:
            self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates


    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, manifold={self.man}, momentum={self.momentum}, eps={self.eps})"