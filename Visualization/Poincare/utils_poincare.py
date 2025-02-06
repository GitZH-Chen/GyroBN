import torch as th
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Visualization.Poincare.GyroBNH import GyroBNH
from frechetmean import Poincare

# Function to generate random points inside the Poincaré ball
def generate_random_poincare_vectors(bs, dim):
    """
    Generate random points inside the Poincaré ball (within unit sphere).
    """
    vectors = np.random.randn(bs, dim)  # Generate random Gaussian vectors
    vectors =  vectors + 1.5
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    radii = np.random.uniform(0, 1, size=(bs, 1))  # Generate random radii
    points = (vectors / norms) * radii  # Scale vectors to have norm < 1 (inside unit ball)
    return points

# Function to plot the Poincaré ball boundary (scatter with black dots)
def plot_poincare_boundary(ax, font_size=12, num=40):
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

# Function to plot the original and centered hyperbolic data
def plot_bn_gyro_poincare(axs, input_poincare, centered_poincare, batch_mean, font_size=12, sample_size=5,
                          mean_size=100, mean_marker='*', grid_color=(0.9, 0.9, 0.9, 0.5)):
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
    axs[0].scatter(input_poincare[:, 0], input_poincare[:, 1], input_poincare[:, 2],
                   s=sample_size, color='blue', label='Input samples', zorder=1)  # Lower zorder
    axs[0].scatter(batch_mean[0], batch_mean[1], batch_mean[2], color='red', s=mean_size,
                   marker=mean_marker, label='Batch mean', zorder=2)  # Higher zorder for batch mean

    plot_poincare_boundary(axs[0], font_size=font_size)
    axs[0].legend(fontsize=font_size)

    # Subplot 2: Centered Poincaré
    axs[1].scatter(centered_poincare[:, 0], centered_poincare[:, 1], centered_poincare[:, 2],
                   s=sample_size, color='green', label='Normalized samples', zorder=1)  # Lower zorder
    axs[1].scatter(0, 0, 0, color='cyan', s=mean_size, marker=mean_marker,
                   label='Resulting mean (0,0,0)', zorder=2)  # Higher zorder for resulting mean

    plot_poincare_boundary(axs[1], font_size=font_size)
    axs[1].legend(fontsize=font_size)

    for ax in axs:
        ax.xaxis._axinfo["grid"]['color'] = grid_color
        ax.yaxis._axinfo["grid"]['color'] = grid_color
        ax.zaxis._axinfo["grid"]['color'] = grid_color


def visualize_gyrobn_poincare(axs, input_poincare_torch, font_size=12, sample_size=5, mean_size=100, mean_marker='*', karcher_steps=1):
    """
    Visualize GyroBN on the Poincaré ball model.

    Parameters:
    - axs: A list of two axes where the visualizations will be drawn.
    - input_poincare_torch: The Poincaré vectors in torch format.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    - mean_marker: Marker style for the mean point.
    - karcher_steps: Number of Karcher steps used in the GyroBN implementation.
    """

    # Initialize GyroBN for Poincaré ball (GyroBNH for hyperbolic space)
    gyro_poincare = GyroBNH(dim=3, manifold=Poincare()).to(th.double)
    centered_poincare_torch, batch_mean_poincare_torch = gyro_poincare(input_poincare_torch)

    # Convert to numpy
    batch_mean_poincare = batch_mean_poincare_torch.detach().numpy()
    centered_poincare = centered_poincare_torch.detach().numpy()

    # Plot Poincaré ball samples and normalized points
    plot_bn_gyro_poincare(axs, input_poincare_torch.detach().numpy(), centered_poincare, batch_mean_poincare,
                          font_size=font_size, sample_size=sample_size, mean_size=mean_size, mean_marker=mean_marker)
