"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Geometry.constantcurvature as manifolds
from RieNets.hnns.layers.att_layers import GraphAttentionLayer
import RieNets.hnns.layers.hyp_layers as hyp_layers
from RieNets.hnns.layers.layers import GraphConvolution, Linear, get_dim_act
from RieNets.hnns.layers.hyp_layers import HypLinear,HypAct

from RieNets.hnns.layers.RBNH import RBNH
from RieNets.hnns.layers.GyroBNH import GyroBNH
from frechetmean import Poincare


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


# class HNN(Encoder):
#     """
#     Hyperbolic Neural Networks.
#     """
#
#     def __init__(self, c, args):
#         super(HNN, self).__init__(c)
#         self.manifold = getattr(manifolds, args.manifold)()
#         assert args.num_layers > 1
#         dims, acts, _ = hyp_layers.get_dim_act_curv(args)
#         hnn_layers = []
#
#         for i in range(len(dims) - 1):
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             hnn_layers.append(HypLinear(self.manifold, in_dim, out_dim, self.c, args.dropout, args.bias))
#             # hnn_layers.append(HypAct(self.manifold, self.c, self.c, act))
#             if args.act:
#                 hnn_layers.append(HypAct(self.manifold, self.c, self.c, act))
#
#         self.layers = nn.Sequential(*hnn_layers)
#         self.encode_graph = False
#
#     def encode(self, x, adj):
#         x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
#         return super(HNN, self).encode(x_hyp, adj)

class HNN(Encoder):
    """
    Hyperbolic Neural Networks with RBN
    """

    def __init__(self, c, args):
        super(__class__, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []

        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(HypLinear(self.manifold, in_dim, out_dim, self.c, args.dropout, args.bias))
            if args.is_bn:
                if args.bn_type=='RBNH':
                    hnn_layers.append(RBNH(out_dim,manifold=Poincare()))
                elif args.bn_type=='GyroBNH':
                    hnn_layers.append(GyroBNH(out_dim, manifold=Poincare()))

            if args.act:
                hnn_layers.append(HypAct(self.manifold, self.c, self.c, act))

        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(__class__, self).encode(x_hyp, adj)

