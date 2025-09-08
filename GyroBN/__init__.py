"""
@author: Ziheng Chen
Please cite the papers below if you use the code:

Ziheng Chen, et al., Gyrogroup Batch Normalization ICLR 2024
Ziheng Chen, et al., Riemannian Batch Normalization: A Gyro Approach 2025

And also the following w.r.t. the special LieBN cases:

Ziheng Chen, et al., A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024.
Ziheng Chen, et al., Adaptive Log-Euclidean metrics for SPD matrix learning. TIP 2024.
Ziheng Chen, et al., LieBN: Batch Normalization over Lie Groups.

Copyright (C) 2025 Ziheng Chen
All rights reserved.
"""

from .GyroBNBase import GyroBNBase
from .GyroBNCCS import GyroBNCCS
from .GyroBNGr import GyroBNGr
from .GyroBNCor import GyroBNCor