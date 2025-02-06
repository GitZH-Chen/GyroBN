import torch as th
import numpy as np
from torch.autograd import Function as F


def trace(A):
    """"
    compute the batch trace of A [...,n,n]
    """
    # trace_vec = th.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
    r_trace = th.einsum("...ii->...", A)
    return r_trace

def inner_product(A, B):
    """"
    compute the batch inner product of A and B, with [...,n,n] [...,n,n]
    """
    r_inner_prod = th.einsum("...ij,...ij->...", A, B)
    return r_inner_prod


def tril_half_diag(A):
    """"[...n,n] A, strictly lower part + 1/2 * half of diagonal part"""
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    half_diag_A = str_tril_A + 0.5 * th.diag_embed(diag_A_vec)
    return half_diag_A

class Lyapunov_eig_solver(F):
    """
    Solving the Lyapunov Equation of BX+XB=C by eigen decomposition
    input (...,n,n) SPD B and symmetric C
    """
    @staticmethod
    def forward(ctx,B,C):
        X, U, L=Ly_forward(B,C)
        ctx.save_for_backward(X, U, L)
        return X
    @staticmethod
    def backward(ctx,dx):
        X, U, L,=ctx.saved_variables
        return Ly_backward(X, U, L, dx)

def Ly_forward(B,C):
    U, S, _ = th.svd(B)
    # SS = S[..., None].repeat(1, 1, S.shape[-1])
    # L = 1. / (SS + SS.transpose(-1,-2))
    L = 1. / (S[..., :, None] + S[..., None, :])
    L[L == -np.inf] = 0; L[L == np.inf] = 0; L[th.isnan(L)] = 0
    X = first_dirivative(U,L,C)
    return X, U, L

def Ly_backward(X, U, L, dx):
    ''''
    dx should be symmetrized
    '''
    sym_dx = sym(dx)
    dc = first_dirivative(U,L,sym_dx)
    tmp = -X.matmul(dc)
    db = tmp+tmp.transpose(-1,-2)
    # if th.norm(db)>20:
    #     pass
    return db, dc

def first_dirivative(U,L,V):
    ''''
    (...,N,N) U, L ,V
    '''
    V_tmp = L * (U.transpose(-1, -2).matmul(V).matmul(U))
    V_New = U.matmul(V_tmp).matmul(U.transpose(-1, -2))
    return V_New

def sym(A):
    ''''
    Make a square matrix symmetrized, (A+A')/2
    '''
    return (A+A.transpose(-1,-2))/2.