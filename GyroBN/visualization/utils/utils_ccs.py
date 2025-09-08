import torch as th
import numpy as np

from GyroBN import GyroBNCCS

from .utils_hyperboloid import plot_hyperboloid,permute_xyz
from .utils_sphere import plot_sphere

def plot_ball(ax, font_size=12, num=40):
    """
    Plot the boundary for the Poincaré ball model using scatter (black dots).
    """
    theta = np.linspace(0, 2 * np.pi, num)
    phi = np.linspace(0, np.pi, num)
    x = np.outer(np.sin(phi), np.cos(theta)).flatten()
    y = np.outer(np.sin(phi), np.sin(theta)).flatten()
    z = np.outer(np.cos(phi), np.ones_like(theta)).flatten()

    # Scatter plot for boundary (black dots)
    ax.scatter(x, y, z, s=1, c='grey', marker='.')

    # Set axis labels using LaTeX format
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=font_size)
    ax.set_ylabel(r'$\mathbf{y}$', fontsize=font_size)
    ax.set_zlabel(r'$\mathbf{z}$', fontsize=font_size)

    # Set less dense ticks
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

def plot_ccs_boundary(ax, font_size=12, model='Stereographic',K=-1):
    if K<0:
        if model in ['Stereographic', 'Klein']:
            plot_ball(ax, font_size=font_size)
        elif model == 'Hyperboloid':
            plot_hyperboloid(ax, font_size=font_size)
    else:
        if model == 'Sphere':
            plot_sphere(ax, font_size=font_size)
    ax.set_xlabel(r'$\mathbf{x}$', fontsize=font_size)
    ax.set_ylabel(r'$\mathbf{y}$', fontsize=font_size)
    ax.set_zlabel(r'$\mathbf{z}$', fontsize=font_size)


# Function to plot the original and centered hyperbolic data
def plot_bn_gyro_ccs(axs, input, centered, batch_mean, origin,
                     model, K,
                     font_size=12, sample_size=5,mean_size=100, mean_marker='*', grid_color=(0.9, 0.9, 0.9, 0.5)):
    """
    Plot Batch Normalized Poincaré ball points on the provided axes.

    Parameters:
    - axs: A tuple of two axis objects where the plots will be drawn.
    - input_poincare: The original Poincaré samples to be visualized.
    - centered_poincare: The centered Poincaré samples after BN.
    - batch_mean: The batch mean of the Poincaré samples.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    - mean_marker: Marker style for the mean point.
    """

    # Subplot 1: Input Poincaré and Batch Mean
    axs[0].scatter(input[:, 0], input[:, 1], input[:, 2],
                   s=sample_size, color='blue', label='Input samples', zorder=1)  # Lower zorder
    axs[0].scatter(batch_mean[0], batch_mean[1], batch_mean[2], color='red', s=mean_size,
                   marker=mean_marker, label='Input mean', zorder=2)  # Higher zorder for batch mean
    axs[0].legend(fontsize=font_size)

    # Subplot 2: Centered Poincaré
    axs[1].scatter(centered[:, 0], centered[:, 1], centered[:, 2],
                   s=sample_size, color='green', label='Output samples', zorder=1)  # Lower zorder
    axs[1].scatter(origin[0], origin[1], origin[2], color='cyan', s=mean_size, marker=mean_marker,
                   label='Output mean', zorder=2)  # Higher zorder for resulting mean
    axs[1].legend(fontsize=font_size)

    plot_ccs_boundary(axs[0], font_size=font_size, model=model, K=K)
    plot_ccs_boundary(axs[1], font_size=font_size, model=model,K=K)

    for ax in axs:
        ax.xaxis._axinfo["grid"]['color'] = grid_color
        ax.yaxis._axinfo["grid"]['color'] = grid_color
        ax.zaxis._axinfo["grid"]['color'] = grid_color


def visualize_gyrobn_ccs(axs, input_tensor, model, K, font_size=12, sample_size=5, mean_size=100, mean_marker='*', max_iter=1000):
    """
    Visualize GyroBN on the Poincaré ball model.

    Parameters:
    - axs: A list of two axes where the visualizations will be drawn.
    - input_tensor: The Poincaré vectors in torch format.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    - mean_marker: Marker style for the mean point.
    - max_iter: Number of iteration steps used in the GyroBN implementation.
    """
    if model in ['Hyperboloid', 'Sphere']:
        dim=2
    else:
        dim = 3
    gyrobn = GyroBNCCS(shape=[dim], model=model, K=K, max_iter=max_iter).to(th.double)
    with th.no_grad():
        if model in ['Sphere']:
            gyrobn.shift.data.fill_(0.4)
        elif model == 'Stereographic' and K > 0:
            gyrobn.shift.data.fill_(1.)
        else:
            gyrobn.shift.data.fill_(0.7)
    centered_tensor = gyrobn(input_tensor)


    input = input_tensor.detach().numpy()
    batch_mean = gyrobn.manifold.frechet_mean(input_tensor).detach().numpy()
    centered = centered_tensor.detach().numpy()
    # origin = gyrobn.manifold.zero(3).numpy()
    origin = gyrobn.manifold.frechet_mean(centered_tensor).detach().numpy()
    if model in ['Hyperboloid', 'Sphere']:
        input = permute_xyz(input)
        batch_mean = permute_xyz(batch_mean)
        centered = permute_xyz(centered)
        origin = permute_xyz(origin)

    plot_bn_gyro_ccs(axs, input, centered, batch_mean,origin,
                     model, K,
                     font_size=font_size, sample_size=sample_size, mean_size=mean_size, mean_marker=mean_marker)
