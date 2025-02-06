"""
    Official Implementation of the SPD matrices presented in
    @inproceedings{chen2024rmlr,
        title={{RMLR}: Extending Multinomial Logistic Regression into General Geometries},
        author={Ziheng Chen and Yue Song and Rui Wang and Xiaojun Wu and Nicu Sebe},
        booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
        year={2024},
    }
"""
import torch as th
import torch.nn as nn

from .sym_functionals import sym_powm,sym_logm,sym_invm,sym_sqrtm,sym_Glogm
from Geometry.spd.functional import inner_product,trace,tril_half_diag,Lyapunov_eig_solver

tril_param_metric = {'AIM','PEM'}
bi_param_metric = {'LEM','ALEM'}
single_param_metric = {'LCM','BWM','PCM','BWCM'}

class SPDMatrices(nn.Module):
    """Computation for SPD data with [...,n,n]"""
    def __init__(self, n,power=1.):
        super().__init__()
        self.n=n; self.dim = int(n * (n + 1) / 2)
        self.register_buffer('power', th.tensor(power))
        self.register_buffer('I', th.eye(n))

        if power == 0:
            raise Exception('power should not be zero with power={:.4f}'.format(power))
        self.sgn_power = -1 if self.power < 0 else 1

    def spd_pow(self, S,power):
        """ computing S^{\theta}"""
        if power == 2.:
            Power_S = S.matmul(S)
        elif power == 1.:
            Power_S = S
        else:
            Power_S = sym_powm.apply(S, power)
        return Power_S

    def RMLR(self, S, P, A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        """
        raise NotImplementedError

class SPDOnInvariantMetric(SPDMatrices):
    """
    Computation for SPD data with [b,c,n,n], the base class of (\theta,\alpha,\beta)-EM/AIM and (\alpha,\beta)-LEM/ALEM
    """
    def __init__(self, n, alpha=1.0, beta=0.,power=1.):
        super(__class__, self).__init__(n,power)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.alpha = alpha;self.beta = beta;
        self.p = (self.alpha + self.n * self.beta) ** 0.5
        self.q = self.alpha ** 0.5

    def alpha_beta_Euc_inner_product(self, tangent_vector1, tangent_vector2):
        """"computing the O(n)-invariant Euclidean inner product"""
        if self.alpha==1. and self.beta==0.:
            X_new = inner_product(tangent_vector1, tangent_vector2)
        else:
            item1 = inner_product(tangent_vector1, tangent_vector2)
            trace_vec1 = trace(tangent_vector1)
            trace_vec2 = trace(tangent_vector2)
            item2 = th.mul(trace_vec1, trace_vec2)
            X_new = self.alpha * item1 + self.beta * item2
        return X_new

class SPDLogEuclideanMetric(SPDOnInvariantMetric):
    """ (\alpha,\beta)-LEM """
    def __init__(self,n,alpha=1.0, beta=0.):
        super(__class__, self).__init__(n,alpha, beta)

    def RMLR(self, S, P, A):
        P_phi = sym_logm.apply(P)
        S_phi = sym_logm.apply(S)
        X_new = self.alpha_beta_Euc_inner_product(S_phi - P_phi, A)

        return X_new

class SPDAdaptiveLogEuclideanMetric(SPDOnInvariantMetric):
    """ (\alpha,\beta)-ALEM """
    def __init__(self,n,alpha=1.0, beta=0.):
        super(__class__, self).__init__(n,alpha, beta)

    def RMLR(self, S, P, A, weight):
        P_phi = sym_Glogm.apply(P, weight)
        S_phi = sym_Glogm.apply(S, weight)
        X_new = self.alpha_beta_Euc_inner_product(S_phi - P_phi, A)

        return X_new

class SPDAffineInvariantMetric(SPDOnInvariantMetric):
    """ Three parameters Affine Invariant Metrics """
    def __init__(self, n, alpha=1.0, beta=0.,power=1.0):
        super(__class__, self).__init__(n,alpha, beta,power)

    def RMLR(self,S,P,A):

        Power_S = self.spd_pow(S, self.power)
        invSquare_power_P = self.spd_pow(P, -self.power / 2)
        in_log = invSquare_power_P.matmul(Power_S).matmul(invSquare_power_P)
        log_data = sym_logm.apply(in_log)

        # computing inner product
        X_new = (1/self.power) *self.alpha_beta_Euc_inner_product(log_data, A)
        return X_new

class SPDLogCholeskyMetric(SPDMatrices):
    """ \theta-LCM """
    def __init__(self, n,power=1.):
        super(__class__, self).__init__(n,power)

    def RMLR(self, S, P, A):
        Power_S = self.spd_pow(S,self.power)
        Power_P = self.spd_pow(P,self.power)

        Chol_of_Power_S = th.linalg.cholesky(Power_S)
        Chol_of_Power_P = th.linalg.cholesky(Power_P)

        item1_diag_vec = th.log(th.diagonal(Chol_of_Power_S, dim1=-2, dim2=-1)) - th.log(th.diagonal(Chol_of_Power_P, dim1=-2, dim2=-1))
        item1 = Chol_of_Power_S.tril(-1) - Chol_of_Power_P.tril(-1) + th.diag_embed(item1_diag_vec)
        X_new = (1 / self.power) * inner_product(item1, tril_half_diag(A))

        return X_new


class SPDBuresWassersteinMetric(SPDOnInvariantMetric):
    """ 2\theta-BWM """
    def __init__(self, n,power=0.5):
        super(__class__, self).__init__(n,power=power)

    def Log(self, point, base_point,power=0.5,omitting_factor=False):
        """
        (PX)^{1/2} = P^{1/2} (P^{1/2} X P^{1/2}) P^{-1/2}
        if omitting_factor = True, omit the factor 1/(|2\theta|)
        [b,c,n,n] point and base_point
        """
        # if self.power==0.5:
        if power == 0.5:
            sqrt_P = sym_sqrtm.apply(base_point)
            # sqrtinv_P = Sqmsym_invm.apply(base_point)
            sqrtinv_P = sym_invm.apply(sqrt_P)
            inter_term = sqrt_P.matmul(point).matmul(sqrt_P)
            sqrt_inter_term = sym_sqrtm.apply(inter_term)
            sqrt_PX = sqrt_P.matmul(sqrt_inter_term).matmul(sqrtinv_P)
            log_P_X = sqrt_PX + sqrt_PX.transpose(-1, -2) - 2 * base_point
        else:
            power_P = sym_powm.apply(base_point,power)
            # invpower_P = sym_powm.apply(base_point, -self.power)
            invpower_P = sym_invm.apply(power_P)
            squarepower_S = sym_powm.apply(point, 2*power)
            inter_term = power_P.matmul(squarepower_S).matmul(power_P)
            sqrt_inter_term = sym_sqrtm.apply(inter_term)
            sqrt_power_2theta_PS = power_P.matmul(sqrt_inter_term).matmul(invpower_P)
            log_P_X = sqrt_power_2theta_PS + sqrt_power_2theta_PS.transpose(-1, -2) - 2 * power_P.matmul(power_P)
        if omitting_factor:
            return log_P_X
        else:
            return 1/(2*abs(self.power)) * log_P_X

    def RMLR(self, S, P, A):
        if self.power == 0.5:
            Power_S = S;
            Power_P = P
        else:
            Power_S = self.spd_pow(S, 2 * self.power)
            Power_P = self.spd_pow(P, 2 * self.power)

        log_P_S = self.Log(Power_S,Power_P,power=0.5,omitting_factor=True)

        Chol_of_power_P = th.linalg.cholesky(Power_P)
        LAL_t = Chol_of_power_P.matmul(A).matmul(Chol_of_power_P.transpose(-1, -2))
        item2 = Lyapunov_eig_solver.apply(Power_P, LAL_t)
        X_new = (1 / (4*self.power)) * inner_product(log_P_S, item2)

        return X_new

class SPDEuclideanMetric(SPDOnInvariantMetric):
    """
    Three parameters Euclidean Metrics
    (1,1,0) standard EM, (-1,1,0) Inverse-Euclidean, (theta,1,0) power Euclidean, (theta,alpha,beta) with theta to 0 is (alpha,beta)-LEM
     """
    def __init__(self,n,alpha=1.0, beta=0.,power=1.0):
        super(SPDEuclideanMetric, self).__init__(n,alpha, beta,power)

    def RMLR(self,S,P,A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        """
        P_power=self.spd_pow(P,self.power)
        S_power = self.spd_pow(S, self.power)

        item1 = (S_power - P_power)
        X_new = 1/self.power * self.alpha_beta_Euc_inner_product(item1,A)
        return X_new

class SPDPowerCholeskyMetric(SPDMatrices):
    """ \theta-PCM: We have not yet implement the matrix-power-deformed version"""
    def __init__(self, n,power=1.):
        super(__class__, self).__init__(n,power)
        # currently, we do not further consider the matrix power deformation
    def RMLR(self, S, P, A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [...,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        """
        Chol_of_S = th.linalg.cholesky(S)
        Chol_of_P = th.linalg.cholesky(P)

        item1_diag_vec = (1/(2*self.power)) * self.diag_function(Chol_of_S,mode='pow',power=self.power) - self.diag_function(Chol_of_P,mode='pow',power=self.power) # K^theta-L^\theta
        tril_item1 = Chol_of_S.tril(-1) - Chol_of_P.tril(-1) + th.diag_embed(item1_diag_vec)

        X_new = inner_product(tril_item1+th.diag_embed(item1_diag_vec), A.tril())

        return X_new

class SPDBuresWassersteinCholeskyMetric(SPDMatrices):
    """ (\theta,M)-BWCM: We have not yet implement the matrix-power-deformed version """
    def __init__(self, n,power=1.):
        super(__class__, self).__init__(n,power)
        # currently, we do not further consider the matrix power deformation
    def RMLR(self, S, P, A, M=None):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [...,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        M: None for standard BWM, and \in D^+_{n} for GBWM
        """
        Chol_of_S = th.linalg.cholesky(S)
        Chol_of_P = th.linalg.cholesky(P)

        item1_diag_vec = (1/(4*self.power)) * self.diag_function(Chol_of_S,mode='pow',power=self.power/2) - self.diag_function(Chol_of_P,mode='pow',power=self.power/2) # K^theta-L^\theta
        if M == None:
            pass
        else:
            item1_diag_vec = item1_diag_vec.div(M)

        tril_item1 = Chol_of_S.tril(-1) - Chol_of_P.tril(-1) + th.diag_embed(item1_diag_vec)

        X_new = inner_product(tril_item1+th.diag_embed(item1_diag_vec), A.tril())

        return X_new
