"""Array manipulation utility functions."""

import numpy as np
from qstack.tools import slice_generator


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
    """Divide numpy arrays avoiding division by zero.

    Args:
        a (numpy.ndarray): Numerator array.
        b (numpy.ndarray): Denominator array.

    Returns:
        numpy.ndarray: Result of element-wise division of a by b, with zeros where b is zero.
    """
    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)


def stack_padding(xs):
    """Stack arrays with different shapes along a new axis by padding smaller arrays with zeros.

    Analogous to numpy.stack(axis=0).

    Args:
        xs (list): List of numpy arrays to be stacked.

    Returns:
        numpy.ndarray : A stacked array with shape (len(xs), *max_shape).

    Raises:
        ValueError: If input arrays have different number of dimensions.
    """
    xs = [np.asarray(x) for x in xs]
    if len({x.ndim for x in xs}) > 1:
        raise ValueError("All input arrays must have the same number of dimensions.")
    shapes = [x.shape for x in xs]
    max_size = max(shapes)
    if max_size == min(shapes):
        return np.stack(xs, axis=0)
    X = np.zeros((len(xs), *max_size))
    for i, x in enumerate(xs):
        slices = tuple(np.s_[0:s] for s in x.shape)
        X[i][slices] = x
    return X


def vstack_padding(xs):
    """Vertically stack arrays with different shapes by padding smaller arrays with zeros.

    1D input arrays of shape (N,) are reshaped to (1,N).
    Analogous to numpy.vstack.

    Args:
        xs (list): List of numpy arrays to be stacked.

    Returns:
        numpy.ndarray : A stacked array with shape (sum(x.shape[0], *max_shape[1:]).

    Raises:
        ValueError: If input arrays have different number of dimensions.
    """
    xs = [np.atleast_2d(np.asarray(x)) for x in xs]
    if len({x.ndim for x in xs}) > 1:
        raise ValueError("All input arrays must have the same number of dimensions.")
    shapes_common_axis, shapes_other_axes = np.split(np.array([x.shape for x in xs]), (1,), axis=1)
    if len(np.unique(shapes_other_axes, axis=0))==1:
        return np.vstack(xs)
    X = np.zeros((shapes_common_axis.sum(), *shapes_other_axes.max(axis=0)))
    for x, s0 in slice_generator(xs, inc=lambda x: x.shape[0]):
        slices = (s0, *(np.s_[0:s] for s in x.shape[1:]))
        X[slices] = x
    return X
