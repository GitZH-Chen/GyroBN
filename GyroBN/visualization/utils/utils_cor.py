import numpy as np
import torch as th

from GyroBN import GyroBNCor
from GyroBN.Geometry import Stereographic
def covariance_to_correlation(cov_matrices):
    # Extract the diagonal elements (variances) from the covariance matrix
    diag_elements = th.diagonal(cov_matrices, dim1=-2, dim2=-1)
    # Compute the standard deviations (sqrt of the diagonal elements)
    std_devs = th.sqrt(diag_elements)
    # Outer product of standard deviations to form the normalization matrix
    normalization_matrix = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
    # Avoid division by zero in case of any zero variances (though variances are typically positive)
    normalization_matrix = th.where(normalization_matrix == 0, th.ones_like(normalization_matrix),
                                    normalization_matrix)
    # Compute the correlation matrix by dividing element-wise
    correlation_matrices = cov_matrices / normalization_matrix
    return correlation_matrices

def random_cor(*shape, eps=1e-6):
    """ Generate random SPD matrices based on the given shape [..., n, n]."""
    assert len(shape) >= 2 and shape[-2] == shape[-1], "Shape must be [..., n, n] for square matrices"
    mu, sigma = -1, 1.5
    n = shape[-1]
    A = th.randn(shape) * sigma + mu
    spd_matrices = th.matmul(A, A.transpose(-2, -1)) + eps * th.eye(n, device=A.device)
    return covariance_to_correlation(spd_matrices)
    # mu, sigma = 0.25, 0.4
    # tril = th.randn(shape).tril(-1) * sigma + mu  # 生成随机矩阵
    # cor= CorEuclideanCholeskyMetric(n=shape[-1])
    # return cor.inv_deformation(tril)



def generate_correlation_boundary(num_matrices_per_case=200):
    """
    Generates boundary points for 3x3 correlation matrices with specific eigenvalue patterns.

    Parameters:
    num_matrices_per_case (int): Number of random correlation matrices to generate for each eigenvalue pattern.

    Returns:
    numpy.ndarray: Array of shape (3, n) containing the lower triangular parts of the boundary points,
                   where n is the total number of generated matrices (3 cases * num_matrices_per_case).
    """

    # Initialize list to store the lower triangular parts of semi-definite correlation matrices
    boundary_parts = []

    # Function to generate a random orthogonal matrix
    def generate_orthogonal_matrix():
        Q, _ = np.linalg.qr(np.random.randn(3, 3))  # QR decomposition to get an orthogonal matrix
        return Q

    # Case 1: Eigenvalues [non-zero, non-zero, zero] (1 zero eigenvalue)
    for _ in range(num_matrices_per_case):
        eigenvalues = np.abs(np.random.rand(3))  # Generate random positive eigenvalues
        eigenvalues[2] = 0  # Ensure one eigenvalue is zero
        perms_case1 = np.unique(np.array([eigenvalues[[i, j, k]]
                                          for i in range(3) for j in range(3) for k in range(3)
                                          if i != j and j != k and i != k]), axis=0)

        for perm in perms_case1:
            Q = generate_orthogonal_matrix()  # Generate random orthogonal matrix
            SPD = Q @ np.diag(perm) @ Q.T  # Semi-definite matrix

            # Normalize to turn it into a correlation matrix
            D = np.diag(1.0 / np.sqrt(np.diag(SPD) + np.finfo(float).eps))  # Add eps to avoid division by zero
            C = D @ SPD @ D  # Correlation matrix

            # Extract the lower triangular part
            lower_part = [C[1, 0], C[2, 0], C[2, 1]]
            boundary_parts.append(lower_part)

    return np.array(boundary_parts).T

def plot_bn_cor(axs, input, boundary, max_bound, font_size=12, sample_size=2, mean_size=100,mean_marker='*'):
    """
    Plot Batch Normalized SPD matrices on the provided axes.

    Parameters:
    - axs: A tuple of two axis objects where the plots will be drawn.
    - spd_tensor: The SPD matrices to be visualized.
    - spd_boundary: The boundary for SPD matrices.
    - metric: The metric used in LieBatchNormSPD ('AIM' or 'RIM').
    - max_bound: Maximum bound for the SPD matrices.
    - font_size: Font size for plot elements.
    - sample_size: Size of the sample points.
    - mean_size: Size of the mean point.
    """
    gyrobn = GyroBNCor(shape=[3,3]).to(th.double)
    pball = Stereographic(K=-1.)
    gyrobn.blocks[0].shift.data.fill_(0.4)
    gyrobn.blocks[1].shift.data.fill_(0.4)

    xs = gyrobn.manifold.to_poly_blocks(input)
    xs_bn = [pball.frechet_mean(x) for x in xs]
    batch_mean = gyrobn.manifold.from_poly_blocks(xs_bn).detach()

    centered_data = gyrobn(input).detach()
    centered_data_s = gyrobn.manifold.to_poly_blocks(centered_data)
    centered_data_s_bn = [pball.frechet_mean(x) for x in centered_data_s]
    origin = gyrobn.manifold.from_poly_blocks(centered_data_s_bn).detach()
    # origin = gyrobn.manifold.frechet_mean(centered_data).detach()

    # Plot input SPD matrices centered at the batch mean
    visualize_cor_tensors(
        spd_tensors=input,
        boundary_dots=boundary,
        ax=axs[0],
        mean=batch_mean,
        font_size=font_size,
        sample_legend="Input samples",
        mean_legend="Input mean",
        sample_size=sample_size,
        mean_size=mean_size,
        mean_marker=mean_marker,
        mean_color='red',
        color='b',
        max_bound=max_bound
    )

    # Plot normalized SPD matrices centered at the identity
    visualize_cor_tensors(
        spd_tensors=centered_data,
        boundary_dots=boundary,
        ax=axs[1],
        mean=origin,
        font_size=font_size,
        sample_legend="Output samples",
        mean_legend="Output mean",
        sample_size=sample_size,
        mean_size=mean_size,
        mean_marker=mean_marker,
        mean_color='cyan',
        color='g',
        max_bound=max_bound
    )


def visualize_cor_tensors(spd_tensors, boundary_dots, ax, mean=None, font_size=10, sample_legend="Samples",
                           mean_legend="Mean", sample_size=5, mean_size=100, color='r',mean_marker='*',
                           mean_color='red', grid_color=(0.9, 0.9, 0.9, 0.5),max_bound=3):
    """
    Visualize SPD(2) matrices in R^3 using the lower triangular part.

    Args:
        spd_tensors: A batch of SPD(2) matrices as a tensor of shape [n, 2, 2].
        boundary_dots: A tuple of three arrays (x, y, z) representing the dots on the SPD boundary surface.
        title: Title for the plot.
        ax: The axis handle where the plot will be drawn.
        mean: The mean SPD(2) matrix (as an SPD(2) matrix) to be plotted as well, if provided.
        font_size: Font size for the plot elements.
        sample_legend: Legend name for the samples.
        mean_legend: Legend name for the mean SPD(2) matrix.
        sample_size: Size of the sample points.
        mean_size: Size of the mean point.
        color: Color for the sample points.
        grid_color: Color for the grid lines.
    """
    # Unpack the boundary dots
    boundary_X, boundary_Y, boundary_Z = boundary_dots[0],boundary_dots[1],boundary_dots[2]

    # Plot the SPD boundary surface using dots
    ax.scatter(boundary_X, boundary_Y, boundary_Z, s=1, c='black', marker='.', alpha=0.3)

    # Extract the lower triangular part of the SPD(2) matrices
    X = spd_tensors[:, 1, 0].numpy()
    Y = spd_tensors[:, 2, 0].numpy()
    Z = spd_tensors[:, 2, 1].numpy()

    # Plot the SPD(2) points in R^3
    ax.scatter(X, Y, Z, s=sample_size, c=color, marker='o', label=sample_legend,alpha=1)

    # If mean is provided, plot the mean SPD(2) matrix as a large red dot
    if mean is not None:
        mean_X = mean[1, 0].item()
        mean_Y = mean[2, 0].item()
        mean_Z = mean[2, 1].item()
        ax.scatter(mean_X, mean_Y, mean_Z, s=mean_size, color=mean_color, label=mean_legend,marker=mean_marker)
        ax.legend(fontsize=font_size)

    ax.set_xlabel(r'$\mathbf{x}$', fontsize=font_size)
    ax.set_ylabel(r'$\mathbf{y}$', fontsize=font_size)
    ax.set_zlabel(r'$\mathbf{z}$', fontsize=font_size)

    ax.set_xticks(np.arange(0, max_bound + 1, 1))
    ax.set_yticks(np.arange(-max_bound, max_bound + 1, 2))
    ax.set_zticks(np.arange(0, max_bound + 1, 1))
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color

    # Set view angle
    # ax.view_init(elev=30, azim=-60)