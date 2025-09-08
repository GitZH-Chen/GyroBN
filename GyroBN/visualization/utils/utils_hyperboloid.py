import torch as th
import numpy as np

def rand_hyperboloid(
    bs: int,
    dim: int = 2,
    *,
    K: float = -1.0,
    mode: str = "uniform",           # "uniform" | "gaussian"
    spatial_max: float = 2.0,
    uniform_low: float = 0.,
    gaussian_mean: float = 0.,
    gaussian_std: float = 1.,
    device=None,
    dtype=None,
    verbose: bool = False,
):
    """
    采样位于超双曲面 H^n_K (Lorentz 模型) 上的点，
    空间部分 ∈ [-spatial_max, spatial_max]，时间分量自动计算。

    参数
    ----
    bs: 批大小
    dim: 空间维度（总维度 = dim + 1，其中第 0 维为时间分量 t）
    K: 曲率（必须 K < 0）
    mode: "uniform" 或 "gaussian"
    spatial_max: 限制空间坐标范围
    gaussian_mean, gaussian_std: 高斯采样的均值与标准差
    verbose: 是否打印 t 的最大值
    """
    if K >= 0:
        raise ValueError("Only K < 0 supported for hyperboloid model.")
    absK = -K

    # ---- 采样空间部分 ----
    if mode.lower() == "uniform":
        space = th.empty(bs, dim, device=device, dtype=dtype).uniform_(uniform_low, spatial_max)
    elif mode.lower() == "gaussian":
        space = th.randn(bs, dim, device=device, dtype=dtype) * gaussian_std + gaussian_mean
        space = th.clamp(space, -spatial_max, spatial_max)
    else:
        raise ValueError("mode must be 'uniform' or 'gaussian'.")

    # ---- 计算时间分量 ----
    val = (space**2).sum(dim=-1, keepdim=True) + 1.0 / absK
    t = th.sqrt(val.clamp(min=1e-9))  # 保证正根

    # ---- 计算理论最大 t ----
    if verbose:
        t_max = (dim * (spatial_max**2) + 1.0 / absK) ** 0.5
        print(f"[rand_hyperboloid] Max possible t with spatial_max={spatial_max}: {t_max:.6f}")

    return th.cat([t, space], dim=-1)





def plot_hyperboloid(ax, font_size=12, K=-1.0,
                     num_r=160, num_ang=240, rho_max=2.6,
                     axis="z", direction="up"):
    """
    Plot upper-sheet hyperboloid (radius model, K<0) as a smooth surface.
    The 'time' coordinate t can be drawn along x/y/z by permuting axes.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    K : float (negative curvature)
    num_r, num_ang : resolution
    rho_max : visible spatial radius
    axis : which axis to put time coord ('x','y','z')
    direction : 'up' or 'down' (opening direction along chosen axis)
    """
    import numpy as np
    assert K < 0, "Hyperboloid requires K < 0"

    # base parameterization: t=z
    rho = np.linspace(0.0, rho_max, num_r)
    theta = np.linspace(0.0, 2*np.pi, num_ang)
    RHO, TH = np.meshgrid(rho, theta, indexing="ij")

    inv_sqrtK = 1.0 / np.sqrt(abs(K))
    xs = RHO * np.cos(TH)
    ys = RHO * np.sin(TH)
    ts = np.sqrt(inv_sqrtK**2 + RHO**2)
    if direction == "down":
        ts = -ts

    # permute so that t goes to chosen axis
    if axis == "z":
        X, Y, Z = xs, ys, ts
    elif axis == "x":
        X, Y, Z = ts, xs, ys
    elif axis == "y":
        X, Y, Z = xs, ts, ys
    else:
        raise ValueError("axis must be 'x','y' or 'z'")

    ax.plot_surface(X, Y, Z,
                    rstride=max(1, num_r//160),
                    cstride=max(1, num_ang//240),
                    linewidth=0, antialiased=True,
                    edgecolor="none", shade=True,
                    alpha=0.15, color="gray")

    # labels
    ax.set_xlabel(r"$\mathbf{x}$", fontsize=font_size)
    ax.set_ylabel(r"$\mathbf{y}$", fontsize=font_size)
    ax.set_zlabel(r"$\mathbf{z}$", fontsize=font_size)
    ax.set_box_aspect((1,1,1))

def permute_xyz(arr, order=(1, 2, 0), sign=(1, 1, 1)):
    """
    Reorder coordinates for (bs,3) or (3,) arrays/tensors.

    Parameters
    ----------
    arr   : np.ndarray or torch.Tensor, shape (bs,3) or (3,)
    order : tuple of ints, permutation of (0,1,2)
    sign  : tuple of ints, per-axis sign flips (1 or -1)

    Returns
    -------
    Same type as input, permuted and with optional sign flip.
    """
    idx = list(order)

    if isinstance(arr, np.ndarray):
        if arr.ndim == 1 and arr.shape[0] == 3:
            out = arr[idx]
        elif arr.ndim == 2 and arr.shape[1] == 3:
            out = arr[:, idx]
        else:
            raise ValueError(f"Expected shape (3,) or (bs,3), got {arr.shape}")
        return out * np.array(sign)

    elif isinstance(arr, th.Tensor):
        if arr.ndim == 1 and arr.shape[0] == 3:
            out = arr[idx]
        elif arr.ndim == 2 and arr.shape[1] == 3:
            out = arr[:, idx]
        else:
            raise ValueError(f"Expected shape (3,) or (bs,3), got {tuple(arr.shape)}")
        return out * th.tensor(sign, dtype=out.dtype, device=arr.device)

    else:
        raise TypeError(f"Unsupported type: {type(arr)}")

