import torch as th


def create_identity_batch(*shape):
    """
    Create a tensor of identity matrices of shape [..., n, n].
    Each matrix is an n x n identity matrix.

    Args:
    *shape: A sequence of integers specifying the dimensions of the resulting tensor. The last two dimensions should be the size of the identity matrix.

    Returns:
    torch.Tensor: A tensor of identity matrices with shape specified by `shape`.
    """
    # Extract n and p from the shape
    *leading_dims, n, p = shape

    # Ensure p is equal to n for identity matrices
    if p != n:
        raise ValueError("For identity matrices, the last two dimensions must be equal.")

    # Initialize a tensor of zeros with the specified shape
    shape = (*leading_dims, n, n) if leading_dims else (n, n)
    identity = th.zeros(shape)

    # Compute indices for the diagonal elements
    indices = th.arange(n)

    # Fill the diagonal of the n x n matrices with 1s
    identity[..., indices, indices] = 1.0

    return identity

def gr_identity_batch(*shape,dtype=None, device=None):
    """
    Create a tensor of Grassmannian identity matrices of shape [..., n, p], where each matrix's upper part is
    a p x p identity matrix, and the lower part is zeros.
    Returns:
    torch.Tensor: A tensor of Grassmannian identity matrices with shape specified by `shape`.
    """
    # Allow a single tuple/Size, e.g. gr_identity_batch(x.shape)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, th.Size)):
        shape = tuple(shape[0])

    if len(shape) < 2:
        raise ValueError("Provide shape as (..., n, p) or (n, p).")

    *leading_dims, n, p = shape
    if p >= n:
        raise ValueError("Grassmannian Gr(p,n) is trivial/undefined when p >= n.")

    out = th.zeros(*leading_dims, n, p, dtype=dtype, device=device)
    if p > 0:
        idx = th.arange(p, device=out.device)
        out[..., idx, idx] = 1.0
    return out

def aux_svd_logmap(M):
    """ svd with flipped order: M \in \bbR^{q \times q}, need for ONB grassmannian logmap and logmap_id"""
    Q1, S1, R1_vh = th.linalg.svd(M)
    # Add clamp for better numerical stability
    S1_clamped = th.clamp(S1, 0, 1)
    # Reorder the columns to make sure S1 in ascending order
    Q1_ascending = th.flip(Q1, dims=[-1])
    R1_vh_ascending = th.flip(R1_vh, dims=[-2])
    S1_ascending = th.flip(S1_clamped, dims=[-1])
    return Q1_ascending,S1_ascending, R1_vh_ascending


#
# def gr_identity_like(U0):
#     # U0 is the tensor with shape [..., n, p]
#     # p is the size of the identity matrix and also U0's last dimension
#     n,p = U0.size(-2),U0.size(-1)
#
#     # Create an identity matrix of size [p, p]
#     identity_p = th.eye(n,p, device=U0.device,dtype=U0.dtype)
#
#     # Expand dimensions to match U0's shape, except for the last two dimensions
#     repeat_size = U0.shape[:-2] + (1, 1)
#     Q2 = identity_p.repeat(*repeat_size)
#
#     return Q2
