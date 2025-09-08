import numpy as np
def plot_sphere(ax, font_size=12, K=1.0,
                num_u=160, num_v=240,
                axis="z"):
    """
    Plot sphere (radius model, K>0) as a smooth surface.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D
    K : float (positive curvature)
    num_u, num_v : resolution
    axis : which axis to put as "time" (just swaps axes like in hyperboloid)
    """

    assert K > 0, "Sphere requires K > 0"

    R = 1.0 / np.sqrt(K)  # sphere radius

    u = np.linspace(0, np.pi, num_u)       # polar angle
    v = np.linspace(0, 2*np.pi, num_v)     # azimuth
    U, V = np.meshgrid(u, v, indexing="ij")

    xs = R * np.sin(U) * np.cos(V)
    ys = R * np.sin(U) * np.sin(V)
    zs = R * np.cos(U)

    # permute axes if needed
    if axis == "z":
        X, Y, Z = xs, ys, zs
    elif axis == "x":
        X, Y, Z = zs, xs, ys
    elif axis == "y":
        X, Y, Z = xs, zs, ys
    else:
        raise ValueError("axis must be 'x','y' or 'z'")

    ax.plot_surface(X, Y, Z,
                    rstride=max(1, num_u//160),
                    cstride=max(1, num_v//240),
                    linewidth=0, antialiased=True,
                    edgecolor="none", shade=True,
                    alpha=0.1, color="gray")

    # labels
    ax.set_xlabel(r"$\mathbf{x}$", fontsize=font_size)
    ax.set_ylabel(r"$\mathbf{y}$", fontsize=font_size)
    ax.set_zlabel(r"$\mathbf{z}$", fontsize=font_size)
    ax.set_box_aspect((1,1,1))
