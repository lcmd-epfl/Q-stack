import os
import warnings
import numpy as np
import sklearn.metrics.pairwise as _SKLEARN_PAIRWISE
from qstack.regression import __path__ as REGMODULE_PATH


def custom_laplacian_kernel(X, Y, gamma):
  """ Compute Laplacian kernel between X and Y

  .. todo::
      Write the docstring
  """
  assert X.shape[1:] == Y.shape[1:]
  def cdist(X, Y):
      K = np.zeros((len(X),len(Y)))
      for i,x in enumerate(X):
          x = np.array([x] * len(Y))
          d = np.abs(x-Y)
          d = np.sum(d, axis=tuple(range(1, len(d.shape))))
          K[i,:] = d
      return K
  K = -gamma * cdist(X, Y)
  np.exp(K, out=K)
  return K


def custom_C_kernels(kernel_function, return_distance_function=False):
    """

    .. todo::
        Write the docstring
    """
    import ctypes
    import sysconfig
    array_2d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')

    lib_path = REGMODULE_PATH[0]+"/lib/manh"+sysconfig.get_config_var('EXT_SUFFIX')
    if not os.path.isfile(lib_path):
        lib_path = REGMODULE_PATH[0]+"/lib/manh.so"

    try:
        manh = ctypes.cdll.LoadLibrary(lib_path)
    except:
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
            assert len(X[0])==len(Y[0])
            K = np.zeros((len(X),len(Y)))
            dist_func(len(X), len(Y), len(X[0]), X, Y, K)
            return K
        return dist_func_c

    else:
        def kernel_func_c(X, Y, gamma):
            assert len(X[0])==len(Y[0])
            K = np.zeros((len(X),len(Y)))
            dist_func(len(X), len(Y), len(X[0]), X, Y, K)
            K *= -gamma
            np.exp(K, out=K)
            return K
        return kernel_func_c


def dot_kernel_wrapper(x, y, *kargs, **kwargs):
    return _SKLEARN_PAIRWISE.linear_kernel(x, y)


def cosine_similarity_wrapper(x, y, *kargs, **kwargs):
    return _SKLEARN_PAIRWISE.cosine_similarity(x, y)


def local_laplacian_kernel_wrapper(X, Y, gamma):
    """ Wrapper that acts as a generic laplacian kernel function
    It simply decides which kernel implementation to call.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    assert X.shape[1:] == Y.shape[1:]
    assert X.ndim > 1   # do not extend so the behavior is the same for 'L' and 'L_custom_py'

    if X.ndim>2:
        kern = local_kernels_dict['L_custom_py']
    else:
        kern = local_kernels_dict['L_custom_c']
        if kern is None:
            warnings.warn("C module for kernel computation is missing/not working. Falling back to python implementation", RuntimeWarning)
            kern = local_kernels_dict['L_sklearn']

    return kern(X, Y, gamma)


local_kernels_dict = {
        'G'          : _SKLEARN_PAIRWISE.rbf_kernel,
        'G_sklearn'  : _SKLEARN_PAIRWISE.rbf_kernel,
        'G_custom_c' : custom_C_kernels('G'),

        'L'          : local_laplacian_kernel_wrapper,
        'L_sklearn'  : _SKLEARN_PAIRWISE.laplacian_kernel,
        'L_custom_c' : custom_C_kernels('L'),
        'L_custom_py': custom_laplacian_kernel,

        'dot'        : dot_kernel_wrapper,
        'cosine'     : cosine_similarity_wrapper,
        }
# legacy kernel names
local_kernels_dict['myG']     = local_kernels_dict['G_custom_c']
local_kernels_dict['myL']     = local_kernels_dict['L_custom_py']
local_kernels_dict['myLfast'] = local_kernels_dict['L_custom_c']
