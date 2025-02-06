import torch as th
import torch.nn as nn

from RieNets.grnets.nn import ProjMap,OrthMap,GyroTransGr,ProjMLR

from RieNets.grnets.GrBN import GyroGrBN
from RieNets.grnets.GrBN import RBNGr
from RieNets.grnets.GrBN import ManifoldNormGr

class GyroGr(nn.Module):
    """GyroGr
        Since the dimension becomes overly low from the second pooling layer onward,
        we apply the BN layer after the first pooling layer.
        The vanilla GyroGr is introduced in "Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach"
    """
    def __init__(self,args):
        super(__class__, self).__init__()

        self.feature = []

        for i in range(len(args.architecture)):
            dims=args.architecture[i]
            # transformation
            self.feature.append(GyroTransGr(channels=args.channels, n=dims[0], p=args.subspace_dim))
            if (i+1)!=len(args.architecture):
                if args.is_pooling:
                    args.final_dim = self.construct_pooling(dims,args)
            else:
                # pooling
                if args.is_final_pooling:
                    args.final_dim = self.construct_pooling(dims,args)
            if args.is_bn and (i + 1) == 1:
                # BN in the middle layer
                self.construct_rbn(args)

        self.feature = nn.Sequential(*self.feature)
        self.construct_classifier(args)

    def forward(self, x):
        x_gras = self.feature(x)
        y = self.classifier(x_gras)
        return y

    def construct_pooling(self, dims, args):
        padding = 1 if dims[0] % 2 != 0 else 0
        self.feature.append(ProjMap())
        self.feature.append(th.nn.AvgPool2d(2, padding=padding))
        final_dim = (dims[0] + 2 * padding - 2) // 2 + 1
        self.feature.append(OrthMap(subspace_dim=args.subspace_dim))
        return final_dim

    def construct_classifier(self, args):
        final_FrMap_dim = args.architecture[-1][-1]
        final_dim = args.final_dim if args.is_final_pooling and args.is_pooling else final_FrMap_dim
        fc_dim=int(final_dim**2 * args.channels)
        self.classifier = ProjMLR(fc_dim,args.class_num)

    def construct_rbn(self, args):
        BN_shape = [args.channels, int(args.final_dim), args.subspace_dim]
        if args.bn_type == 'RBNGr':
            self.feature.append(RBNGr(shape=BN_shape))
        elif args.bn_type == 'GyroGrBN':
            self.feature.append(GyroGrBN(shape=BN_shape))
        elif args.bn_type == 'ManifoldNormGr':
            self.feature.append(ManifoldNormGr(shape=BN_shape))
        else:
            raise NotImplementedError