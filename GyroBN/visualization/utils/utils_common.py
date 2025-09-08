import numpy as np

def set_axis(axs, xset, yset, zset, font_size=12):
    """
    Set 3D axis limits and ticks from [min, max, step] lists.

    Parameters
    ----------
    axs : single axis OR list of axes
    xset, yset, zset : list [min, max, step]
    """
    if not isinstance(axs, (list, tuple)):
        axs = [axs]

    for ax in axs:
        # X
        ax.set_xlim(xset[0], xset[1])
        ax.set_xticks(np.arange(xset[0], xset[1] + 1e-9, xset[2]))

        # Y
        ax.set_ylim(yset[0], yset[1])
        ax.set_yticks(np.arange(yset[0], yset[1] + 1e-9, yset[2]))

        # Z
        ax.set_zlim(zset[0], zset[1])
        ax.set_zticks(np.arange(zset[0], zset[1] + 1e-9, zset[2]))

        ax.tick_params(labelsize=font_size)
