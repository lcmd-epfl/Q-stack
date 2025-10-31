import numpy as np

def from_tril(mat_tril):
    """Restore a symmetric matrix from its lower-triangular form.

    Args:
        mat_tril (numpy.ndarray): 1D array containing matrix in lower-triangular form.

    Returns:
        numpy.ndarray: 2D symmetric matrix.
    """
    n = int((np.sqrt(1+8*len(mat_tril))-1)/2)
    ind = np.tril_indices(n)
    mat = np.zeros((n,n))
    mat[ind] = mat_tril
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat

def sqrtm(m, eps=1e-13):
    """Compute the matrix square root of a symmetric matrix.

    Args:
        m (numpy.ndarray): Symmetric matrix.
        eps (float): Threshold for eigenvalues to be considered zero. Defaults to 1e-13.

    Returns:
        numpy.ndarray: Symmetrized square root of the matrix.
    """
    e, b = np.linalg.eigh(m)
    e[abs(e) < eps] = 0.0
    sm = b @ np.diag(np.sqrt(e)) @ b.T
    return (sm+sm.T)*0.5
