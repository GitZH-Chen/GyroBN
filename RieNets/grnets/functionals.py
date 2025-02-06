import math
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

def trace(A):
    """"
    compute the batch trace of A [...,n,n]
    """
    # trace_vec = th.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
    r_trace = th.einsum("...ii->...", A)
    return r_trace

def ini_skew_B(channels, n, p, mode='uniform'):
    weight_dim=n-p
    B = th.zeros(channels, weight_dim, p)
    feature_dim = channels * weight_dim * p
    if mode == 'uniform':
        init_matrix_uniform(B, feature_dim)
    elif mode == 'normal':
        B.normal_(mean=0, std=(feature_dim) ** -0.5)
    elif mode == 'std_normal':
        B.normal_(mean=0, std=1)
    else:
        raise NotImplementedError
    return B

def init_matrix_uniform(A,fan_in,factor=6):
    bound = math.sqrt(factor / fan_in) if fan_in > 0 else 0
    nn.init.uniform_(A, -bound, bound)

def patch_len(n, epochs):
    """将特征向量分为epochs个时期，返回一个列表，列表中包含每个时期中特征向量个数"""
    list_len = []
    base = n // epochs
    for i in range(epochs):
        list_len.append(base)
    for i in range(n - base * epochs):
        list_len[i] += 1
    # 验证
    if sum(list_len) == n:
        return list_len
    else:
        return ValueError('check your epochs and axis should be split again')

class OrthmapFunction(F):
    @staticmethod
    def forward(ctx, x, p):
        U, S, V = th.linalg.svd(x)
        ctx.save_for_backward(U, S)
        # if len(x.shape) == 3:
        #     res = U[:, :, :p]
        # else:
        #     res = U[:, :, :, :p]
        res = U[..., :p]
        return res

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b, c, h, w = grad_output.shape
        p = h - w
        pad_zero = th.zeros(b, c, h, p)
        # 调整输出格式
        grad_output = th.cat((grad_output, pad_zero), 3)
        Ut = U.transpose(-1, -2)
        K = calcuK(S)
        # U * (K.T。(U.T * dL/dx)sym) * U.T
        # mid_1 = torch.matmul(Ut, grad_output)
        # mid_2 = K.transpose(-1, -2) * torch.add(mid_1, mid_1.transpose(-1, -2))
        # mid_3 = torch.matmul(U, mid_2)
        # return torch.matmul(mid_3, Ut), None
        mid_1 = K.transpose(-1, -2) * th.matmul(Ut, grad_output)
        mid_2 = th.matmul(U, mid_1)
        return th.matmul(mid_2, Ut), None

def calcuK(S):
    b, c, h = S.shape
    Sr = S.reshape(b, c, 1, h)
    Sc = S.reshape(b, c, h, 1)
    K = Sc - Sr
    K = 1.0 / K
    K[th.isinf(K)] = 0
    return K

