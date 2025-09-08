import torch as th
import torch.nn as nn
from typing import List

from GyroBN import GyroBNBase, GyroBNCCS
from GyroBN.Geometry import CorPolyHyperbolicCholeskyMetric


class GyroBNCor(GyroBNBase):
    """
    GyroBN on the Correlation manifold (Cor+(n)).

    Maps each correlation matrix (via Cholesky factorization) into a product
    of hyperbolic Poincaré balls; applies GyroBN factor-wise; then maps back
    to correlation space.

    Args:
        shape (list[int]): Dimension of correlation matrix input.
        max_iter (int, default=1000): Max iterations for Fréchet mean computations.
        momentum (float, default=0.1): Momentum for running statistics.
        track_running_stats (bool, default=True): Keep running mean/var.
        init_1st_batch (bool, default=False): Initialize running stats by the first batch.
        eps (float, default=1e-6): Numerical stability for variance.

    Note:
        Uses CorPolyPoincaréBallCholeskyMetric to map C ↔ {x_i} (product of Poincaré balls);
        applies per-block hyperbolic GyroBN and maps back.

    References:
        - Ziheng Chen, et al, Gyrogroup Batch Normalization. ICLR 2024
        - Ziheng Chen, et al, Riemannian Batch Normalization: A Gyro Approach. 2025
        - Ziheng Chen, et al, Riemannian Networks over Full-Rank Correlation Matrices. 2025
        - Paul David, et al, A Riemannian Structure for Correlation Matrices. Operators and Matrices 2019
        - Yann Thanwerdas, et al, Theoretically and Computationally Convenient Geometries on Full-Rank Correlation Matrices. SIMAX 2022
    """
    def __init__(self,
                 shape,
                 max_iter: int = 1000,
                 momentum: float = 0.1,
                 track_running_stats: bool = True,
                 init_1st_batch: bool = False,
                 eps: float = 1e-6):
        super().__init__(shape=shape,
                         batchdim=[0],
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         init_1st_batch=init_1st_batch,
                         eps=eps)
        assert shape[-1] == shape[-2]
        self.n = shape[-1]
        self.max_iter=max_iter
        self.get_manifold()
        self.set_h_bn()

    def get_manifold(self):
        self.manifold = CorPolyHyperbolicCholeskyMetric(n=self.n, jitter=self.eps)

    def set_h_bn(self):
        # 为每个块 i=1..n-1 建一个 Poincaré GyroBN（维度为 i）
        blocks: List[nn.Module] = []
        for i in range(1, self.n):
            blocks.append(
                GyroBNCCS(
                    shape=[i],
                    model='Stereographic',
                    K=-1.0,
                    max_iter=self.max_iter,
                    momentum=self.momentum,
                    track_running_stats=self.track_running_stats,
                    init_1st_batch=self.init_1st_batch,
                    eps=self.eps,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, C: th.Tensor) -> th.Tensor:
        # C -> {x_i}
        xs = self.manifold.to_poly_blocks(C)
        xs_bn = [bn(x) for bn, x in zip(self.blocks, xs)]
        # {x_i'} -> C'
        C_out = self.manifold.from_poly_blocks(xs_bn)
        return C_out
