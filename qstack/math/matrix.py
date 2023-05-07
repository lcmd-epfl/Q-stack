import numpy as np

def from_tril(mat_tril):
    """Restore a symmetric matrix from its lower-triangular form.

    Args:
        mat_tril (numpy 1darray): matrix in a lower-triangular form.

    Returns:
        A numpy 2darray containing the matrix.
    """
    n = int((np.sqrt(1+8*len(mat_tril))-1)/2)
    ind = np.tril_indices(n)
    mat = np.zeros((n,n))
    mat[ind] = mat_tril
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat
