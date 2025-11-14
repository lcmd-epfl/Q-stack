"""Local (atomic) kernel implementations.

Provides:
    local_kernels_dict: Dictionary mapping kernel names to their implementations.
"""

import os
import ctypes
import sysconfig
import warnings
import numpy as np
import sklearn.metrics.pairwise as _SKLEARN_PAIRWISE
from qstack.regression import __path__ as REGMODULE_PATH


RAM_BATCHING_SIZE = 1024**3 * 5  # 5GiB
def compute_distance_matrix(R1,R2):
    """Compute the manhattan-distance matrix.

    This computes (||r_1 - r_2||_1) between the samples of R1 and R2,
    using a batched python/numpy implementation,
    designed to be more memory-efficient than a single numpy call and faster than a simple python for loop.

    This function is a batched-over-R1 implementation of the following code:
    `return np.sum( (R1[:,None, ...]-R2[None,:, ...])**2, axis=tuple(range(2, R1.ndim)))`

    Args:
        R1 (numpy ndarray): First set of samples (can be multi-dimensional).
        R2 (numpy ndarray): Second set of samples.

    Returns:
        numpy ndarray: squared-distance matrix of shape (len(R1), len(R2)).

    Raises:
        RuntimeError: If X and Y have incompatible shapes.
    """
    if R1.ndim != R2.ndim or R1.shape[1:] != R2.shape[1:]:
        raise RuntimeError(f'incompatible shapes for R1 ({R1.shape:r}) and R2 ({R2.shape:r})')


    # determine batch size (batch should divide the larger dimention)
    if R1.shape[0] < R2.shape[0]:
        transpose_flag = True
        R2,R1 = R1,R2
    else:
        transpose_flag = False
    dtype=np.result_type(R1,R2)
    out = np.zeros((R1.shape[0], R2.shape[0]), dtype=dtype)

    # possible weirdness: how is the layout of dtype done if dtype.alignment != dtype.itemsize?
    batch_size = int(np.floor(RAM_BATCHING_SIZE/ (dtype.itemsize * np.prod(R2.shape))))

    if batch_size == 0:
        batch_size = 1

    if min(R1.shape[0],R2.shape[0]) == 0 or batch_size >= R1.shape[0]:
        dists = R1[:,None]-R2[None,:]
        #np.pow(dists, 2, out=dists)
        np.abs(dists, out=dists)
        out = np.sum(dists, axis=tuple(range(2,dists.ndim)))
    else:
        n_batches = int(np.ceil(R1.shape[0] / batch_size))
        dists = np.zeros( (batch_size, *R2.shape), dtype=dtype )
        for batch_i in range(n_batches):
            batch_start = batch_i*batch_size
            this_batch_size = min(batch_size, R1.shape[0]-batch_start)

            R1_view = R1[batch_start : batch_start + this_batch_size, None, ...]
            np.subtract(R1_view, R2[None,:], out=dists[:this_batch_size])
            #np.pow(dists[:this_batch_size], 2, out=dists[:this_batch_size])  # For Euclidean distance
            np.abs(dists[:this_batch_size], out=dists[:this_batch_size])
            np.sum(dists[:this_batch_size], out=out[batch_start : batch_start+this_batch_size], axis=tuple(range(2,dists.ndim)))

    if transpose_flag:
        out = out.T
    return out

def custom_laplacian_kernel(X, Y, gamma):
    """Compute Laplacian kernel between X and Y using Python implementation.

    K(x, y) = exp(-gamma * ||x - y||_1)

    Args:
        X (numpy ndarray): First set of samples (can be multi-dimensional).
        Y (numpy ndarray): Second set of samples.
        gamma (float): Kernel width parameter.

    Returns:
        numpy ndarray: Laplacian kernel matrix of shape (len(X), len(Y)).

    Raises:
        RuntimeError: If X and Y have incompatible shapes.
    """
    if X.shape[1:] != Y.shape[1:]:
        raise RuntimeError(f"Incompatible shapes {X.shape} and {Y.shape}")
    K = -gamma * compute_distance_matrix(X,Y)
    np.exp(K, out=K)
    return K


def custom_C_kernels(kernel_function, return_distance_function=False):
    """Create kernel function wrappers using C implementation for speed.

    Args:
        kernel_function (str): Kernel type ('L' for Laplacian, 'G' for Gaussian).
        return_distance_function (bool): If True, returns distance function instead of kernel. Defaults to False.

    Returns:
        callable or None: Kernel or distance function, or None if C library cannot be loaded.
    """
    array_2d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')

    lib_path = REGMODULE_PATH[0]+"/lib/manh"+sysconfig.get_config_var('EXT_SUFFIX')
    if not os.path.isfile(lib_path):
        lib_path = REGMODULE_PATH[0]+"/lib/manh.so"
    try:
        manh = ctypes.cdll.LoadLibrary(lib_path)
    except OSError:
        return None

    if kernel_function == 'L':
        dist_func = manh.manh
    elif kernel_function == 'G':
        dist_func = manh.eucl
    dist_func.restype = ctypes.c_int
    dist_func.argtypes = [
      ctypes.c_int,
      ctypes.c_int,
      ctypes.c_int,
      array_2d_double,
      array_2d_double,
      array_2d_double]

    if return_distance_function:
        def dist_func_c(X, Y):
            if len(X[0])!=len(Y[0]):
                raise RuntimeError(f"Incompatible shapes {X.shape} and {Y.shape}")
            K = np.zeros((len(X),len(Y)))
            dist_func(len(X), len(Y), len(X[0]), X, Y, K)
            return K
        return dist_func_c

    else:
        def kernel_func_c(X, Y, gamma):
            if len(X[0])!=len(Y[0]):
                raise RuntimeError(f"Incompatible shapes {X.shape} and {Y.shape}")
            K = np.zeros((len(X),len(Y)))
            dist_func(len(X), len(Y), len(X[0]), X, Y, K)
            K *= -gamma
            np.exp(K, out=K)
            return K
        return kernel_func_c


def dot_kernel_wrapper(x, y, *_kargs, **_kwargs):
    """Compute linear (dot product) kernel.

    Args:
        x (numpy ndarray): First set of samples.
        y (numpy ndarray): Second set of samples.
        *_kargs: Unused positional arguments (for compatibility).
        **_kwargs: Unused keyword arguments (for compatibility).

    Returns:
        numpy ndarray: Linear kernel matrix.
    """
    return _SKLEARN_PAIRWISE.linear_kernel(x, y)


def cosine_similarity_wrapper(x, y, *_kargs, **_kwargs):
    """Compute cosine similarity kernel.

    Args:
        x (numpy ndarray): First set of samples.
        y (numpy ndarray): Second set of samples.
        *_kargs: Unused positional arguments (for compatibility).
        **_kwargs: Unused keyword arguments (for compatibility).

    Returns:
        numpy ndarray: Cosine similarity matrix.
    """
    return _SKLEARN_PAIRWISE.cosine_similarity(x, y)


def local_laplacian_kernel_wrapper(X, Y, gamma):
    """Decide which kernel implementation to call.

    Wrapper that acts as a generic Laplacian kernel function.

    Args:
        X (numpy ndarray): First set of samples (can be multi-dimensional).
        Y (numpy ndarray): Second set of samples.
        gamma (float): Kernel width parameter.

    Returns:
        numpy ndarray: Laplacian kernel matrix of shape (len(X), len(Y)).

    Raises:
        RuntimeError: If X and Y have incompatible shapes.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    if X.shape[1:] != Y.shape[1:]:
        raise RuntimeError(f"Incompatible shapes {X.shape} and {Y.shape}")
    if X.ndim==1: # do not extend so the behavior is the same for 'L' and 'L_custom_py'
        raise RuntimeError("Dimensionality of X should be > 1")

    if X.ndim>2:
        kern = local_kernels_dict['L_custom_py']
    else:
        kern = local_kernels_dict['L_custom_c']
        if kern is None:
            warnings.warn("C module for kernel computation is missing/not working. Falling back to python implementation", RuntimeWarning, stacklevel=2)
            kern = local_kernels_dict['L_sklearn']

    return kern(X, Y, gamma)


local_kernels_dict = {
        'G'          : _SKLEARN_PAIRWISE.rbf_kernel,
        'L'          : local_laplacian_kernel_wrapper,

        'dot'        : dot_kernel_wrapper,
        'cosine'     : cosine_similarity_wrapper,

        'G_sklearn'  : _SKLEARN_PAIRWISE.rbf_kernel,
        'G_custom_c' : custom_C_kernels('G'),

        'L_sklearn'  : _SKLEARN_PAIRWISE.laplacian_kernel,
        'L_custom_c' : custom_C_kernels('L'),
        'L_custom_py': custom_laplacian_kernel,
        }
# legacy kernel names
local_kernels_dict['myG']     = local_kernels_dict['G_custom_c']
local_kernels_dict['myL']     = local_kernels_dict['L_custom_py']
local_kernels_dict['myLfast'] = local_kernels_dict['L_custom_c']
