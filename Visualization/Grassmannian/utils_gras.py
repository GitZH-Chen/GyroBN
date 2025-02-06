import matplotlib.pyplot as plt
import torch as th
import numpy as np
from Visualization.Grassmannian.GrGroBNImpl4vis import GrGyroBNImpl
# from RieNets.grnets.GrBN.GrGroBNImpl import GyroGrBN

def flip_z_if_negative(vectors):
    # Copy the original array to avoid modifying the input array
    flipped_vectors = vectors.copy()
    if vectors.ndim==2:
        # Find indices where z-coordinate is negative
        if flipped_vectors[0,0] < 0:
            # Flip the sign of z-coordinate for the corresponding indices
            flipped_vectors[0,0] *= -1
    else:
        # Find indices where z-coordinate is negative
        negative_z_indices = flipped_vectors[:, 0,0] < 0
        # Flip the sign of z-coordinate for the corresponding indices
        flipped_vectors[negative_z_indices, 0,0] *= -1
    return flipped_vectors

def plot_bn_gyro_gras(axs, input_grassmannians, centered_grassmannians, batch_mean, font_size=12, sample_size=5,
                      mean_size=100, mean_marker='*',
                      grid_color = (0.9, 0.9, 0.9, 0.5)):
    """
    Plot Batch Normalized Grassmannian points on the provided axes.

    Parameters:
    - axs: A tuple of two axis objects where the plots will be drawn.
    - input_grassmannians: The original Grassmannian samples to be visualized.
    - centered_grassmannians: The centered Grassmannian samples after BN.
    - batch_mean: The batch mean of the Grassmannian samples.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    - mean_marker: Marker style for the mean point.
    """

    # Subplot 1: Input Grassmannians and Batch Mean
    axs[0].scatter(input_grassmannians[:, 0, 0], input_grassmannians[:, 1, 0], input_grassmannians[:, 2, 0],
                   s=sample_size, color='blue', label='Input samples')

    # Plot batch mean with mean_marker
    axs[0].scatter(batch_mean[0], batch_mean[1], batch_mean[2], color='red', s=mean_size,
                   marker=mean_marker, label='Batch mean')

    # Define the mesh in spherical coordinates for visualization
    theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
    phi = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(np.size(theta)), np.cos(phi))

    axs[0].plot_surface(x, y, z, color="gray", alpha=0.3)
    axs[0].set_xlabel(r'$\mathbf{x}$', fontsize=font_size)
    axs[0].set_ylabel(r'$\mathbf{y}$', fontsize=font_size)
    axs[0].set_zlabel(r'$\mathbf{z}$', fontsize=font_size)
    axs[0].tick_params(axis='both', which='major', labelsize=font_size)
    axs[0].legend(fontsize=font_size)

    # Subplot 2: Centered Grassmannians
    axs[1].scatter(centered_grassmannians[:, 0, 0], centered_grassmannians[:, 1, 0], centered_grassmannians[:, 2, 0],
                   s=sample_size, color='green', label='Normalized samples')

    # Plot neutral element with mean_marker
    axs[1].scatter(1, 0, 0, color='cyan', s=mean_size, marker=mean_marker, label='Resulting mean $I_{n,p}$')

    axs[1].plot_surface(x, y, z, color="gray", alpha=0.3)
    axs[1].set_xlabel(r'$\mathbf{x}$', fontsize=font_size)
    axs[1].set_ylabel(r'$\mathbf{y}$', fontsize=font_size)
    axs[1].set_zlabel(r'$\mathbf{z}$', fontsize=font_size)
    axs[1].tick_params(axis='both', which='major', labelsize=font_size)
    axs[1].legend(fontsize=font_size)

    # Set aspect ratio for both subplots
    for ax in axs:
        ax.set_box_aspect([1, 2, 2])  # Equal aspect ratio
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis._axinfo["grid"]['color'] = grid_color
        ax.yaxis._axinfo["grid"]['color'] = grid_color
        ax.zaxis._axinfo["grid"]['color'] = grid_color



# Example usage:
def visualize_gyrobn_gras(axs, input_grassmannians_torch, font_size=12, sample_size=5, mean_size=100, mean_marker='*', karcher_steps=1):
    """
    Visualize GyroBN on Grassmannians.

    Parameters:
    - axs: A list of four axes where the first two will be used for GyroBN visualization.
    - bs: Batch size for the Grassmannian samples.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    - mean_marker: Marker style for the mean point.
    - karcher_steps: Number of Karcher steps used in the GyroBN implementation.
    """

    # Initialize the GrGyroBNImpl module
    gyro = GrGyroBNImpl(shape=[3, 1],karcher_steps=karcher_steps).to(th.double)
    gyro.shift.data=th.tensor(0.2)

    # Step 3: Get the centered Grassmannians and batch mean using GrGyroBNImpl
    centered_grassmannians_torch, batch_mean_torch = gyro(input_grassmannians_torch)
    batch_mean = flip_z_if_negative(batch_mean_torch.detach().numpy())
    centered_grassmannians = flip_z_if_negative(centered_grassmannians_torch.detach().numpy())

    # Step 4: Plot the GyroBN results on the first two subplots
    plot_bn_gyro_gras(axs, input_grassmannians_torch, centered_grassmannians, batch_mean,
                 font_size=font_size, sample_size=sample_size, mean_size=mean_size, mean_marker=mean_marker)

