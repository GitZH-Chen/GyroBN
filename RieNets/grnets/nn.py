import torch as th
import torch.nn as nn

from .functionals import OrthmapFunction, ini_skew_B

from Geometry.Grassmannian.GrGyro import GrassmannianGyro

lim_dim_eigh=32 # bigger than lim_dim_eigh use cpu
# lim_dim_qr=32 # less than lim_dim_eigh use cpu

class GyroTransGr(nn.Module):
    ''' Gyro Translation under ONB'''
    def __init__(self, channels,n,p):
        super(__class__, self).__init__()
        self.channels,self.n,self.p = channels,n,p
        self.Gyro=GrassmannianGyro(n,p)
        self.weight = nn.Parameter(ini_skew_B(self.channels,self.n,self.p))

    def forward(self, X):
        skew_bias = self.Gyro.B2skrew(self.weight)
        X_trans = self.Gyro.left_gyrotranslation_skew2U(skew_bias, X)
        return X_trans

    def __repr__(self):
        return f"{self.__class__.__name__}(channels={self.channels},n={self.n},p={self.p})"

class OrthMap(nn.Module):
    '''
        OrthMap: re-orthogonalization by eigh
        Note that th.linalg.svd(X) will return NaN grad
    '''
    def __init__(self, subspace_dim,mode='eigh'):
        super(__class__, self).__init__()
        self.subspace_dim=subspace_dim;self.mode=mode;self.gpu2cpu=False
    def forward(self,X):
        if X.device != 'cpu' and X.shape[-2]>lim_dim_eigh:
            self.gpu2cpu=True
            device=X.device
            X = X.to('cpu')

        if self.mode=='eigh':
            S, U = th.linalg.eigh(X)
            S_desc, indices = th.sort(S, descending=True)

            # Sort eigenvectors to correspond to sorted eigenvalues
            U_desc = th.gather(U, -1, indices.unsqueeze(-2).expand_as(U))

            output = U_desc[..., :self.subspace_dim]
        elif self.mode =='svd':
            output = OrthmapFunction.apply(X, self.subspace_dim)
        else:
            raise NotImplementedError(f'unknown mode {self.mode}')

        return output.to(device) if self.gpu2cpu else output

    def __repr__(self):
        return f"{self.__class__.__name__}(subspace_dim={self.subspace_dim}, mode={self.mode})"

class ProjMap(nn.Module):
    def forward(self,X):
        return X @ X.transpose(-1, -2)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ProjMLR(nn.Module):
    def __init__(self, fc_dim, classnum):
        super(__class__, self).__init__()
        self.PJ = ProjMap()
        self.linear = nn.Linear(fc_dim, classnum).double()

    def forward(self, x):
        x_vec = self.PJ(x).view(x.shape[0], -1)
        y = self.linear(x_vec)
        return y