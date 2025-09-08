import torch as th
import torch.nn as nn

class GyroBNBase(nn.Module):
    """
    Base class for Gyrogroup Batch Normalization (GyroBN) on manifolds.

    Args:
        shape (list[int]): Input shape excluding batch dimensions (e.g., [d] or [c, d, d]).
        batchdim (list[int], default=[0]): Indices of batch dimensions to normalize over.
        momentum (float, default=0.1): Momentum for updating running statistics.
        track_running_stats (bool, default=True): Keep running mean/var.
        init_1st_batch (bool, default=False): Initialize running stats by the first batch.
        eps (float, default=1e-6): Small value added to variance for stability.

    Note:
        Subclasses must implement: set_parameters(), register_running_stats(), get_manifold().
        Core ops: left gyrotranslation (centering), scalar gyromultiplication (scaling),
        left gyrotranslation (biasing); running mean via binary barycenter.

    References:
        - Ziheng Chen, et al, Gyrogroup Batch Normalization. ICLR 2024
        - Ziheng Chen, et al, Riemannian Batch Normalization: A Gyro Approach. 2025
        - Ioffe, S., et al, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015
        - Ungar, A.A. Analytic Hyperbolic Geometry and Albert Einstein’s Special Theory of Relativity (2nd ed.). World Scientific, 2022
    """
    def __init__(self,shape, batchdim=[0], momentum=0.1,
                 track_running_stats=True, init_1st_batch=False,eps=1e-6):
        super().__init__()
        self.shape=shape;
        self.batchdim=batchdim;
        self.momentum = momentum;
        self.eps=eps
        self.track_running_stats = track_running_stats
        self.init_1st_batch = init_1st_batch
        # Require customized set_parameters, register_running_stats, get_manifold

    def set_parameters(self):
        raise NotImplementedError

    def register_running_stats(self):
        raise NotImplementedError

    def get_manifold(self):
        raise NotImplementedError

    def updating_running_statistics(self, batch_mean, batch_var):
        if self.running_mean is None:
            self.running_mean = batch_mean.detach()
        else:
            self.running_mean = self.manifold.geodesic(self.running_mean, batch_mean.detach(),self.momentum)

        if self.running_var is None:
            self.running_var = batch_var.detach()
        else:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()

    def __repr__(self):
        attributes = []

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, th.Tensor):
                if value.numel() == 1:
                    val_str = f"{value.item():.4f}"
                else:
                    val_str = f"shape={tuple(value.shape)}"
                attributes.append(f"{key}={val_str}")
            else:
                attributes.append(f"{key}={value}")

        for name, buffer in self.named_buffers(recurse=False):
            if buffer.numel() == 1:
                val_str = f"{buffer.item():.4f}"
            else:
                val_str = f"shape={tuple(buffer.shape)}"
            attributes.append(f"{name}={val_str}")

        return f"{self.__class__.__name__}({', '.join(attributes)}) \n"

