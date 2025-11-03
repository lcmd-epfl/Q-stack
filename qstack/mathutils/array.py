import numpy as np


def scatter(values, indices):
    """Scatter values into a new array based on provided indices.

    Does the same as
    ```
    for i, j in enumerate(indices):
        x[...,i,j] = values[...,i]
    ```

    Args:
        values (numpy.ndarray): Array of values to be scattered of shape (..., N).
        indices (numpy.ndarray): Array of indices indicating where to scatter the values of shape (N,).

    Returns:
        numpy.ndarray: New array with scattered values of shape (..., N, max(indices)+1).


    """
    x = np.zeros((*values.shape, max(indices)+1))
    x[...,np.arange(len(indices)),indices] = values
    return x


def safe_divide(a, b):
    """Wrapper for numpy divide that avoids division by zero.

    Args:
        a (numpy.ndarray): Numerator array.
        b (numpy.ndarray): Denominator array.

    Returns:
        numpy.ndarray: Result of element-wise division of a by b, with zeros where b is zero.
    """

    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)


def vstack_padding(xs):
    """Vertically stack arrays with different shapes by padding smaller arrays with zeros.

    Args:
        xs (list): List of numpy arrays to be stacked.

    Returns:
        numpy.ndarray : A stacked array with shape (len(xs), *max_shape).

    Raises:
        ValueError: If input arrays have different number of dimensions.
    """
    if len({x.ndim for x in xs}) > 1:
        raise ValueError("All input arrays must have the same number of dimensions.")
    max_size = max(x.shape for x in xs)
    X = np.zeros((len(xs), *max_size))
    for i, x in enumerate(xs):
        slices = tuple(np.s_[0:s] for s in x.shape)
        X[i][slices] = x
    return X
