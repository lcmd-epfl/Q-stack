import os
import argparse
import warnings
from types import SimpleNamespace
import numpy as np
from .local_kernels import local_kernels_dict
from .global_kernels import global_kernels_dict, get_global_K

REGMODULE_PATH = os.path.dirname(__file__)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, defaults.gdict)
        for value in values:
            key, value = value.split('=')
            for t in [int, float]:
                try:
                    value = t(value)
                    break
                except ValueError:
                    continue
            getattr(namespace, self.dest)[key] = value


defaults = SimpleNamespace(
  sigma=32.0,
  eta=1e-5,
  kernel='L',
  gkernel=None,
  gdict={'alpha':1.0, 'normalize':1, 'verbose':0},
  test_size=0.2,
  n_rep=5,
  splits=5,
  train_size=[0.125, 0.25, 0.5, 0.75, 1.0],
  etaarr=list(np.logspace(-10, 0, 5)),
  sigmaarr=list(np.logspace(0,6, 13)),
  sigmaarr_mult=list(np.logspace(0,2, 5)),
  random_state=0,
  )


def get_local_kernel(arg):
    """ Obtains a local-envronment kernel by name.

    Args:
        arg (str): the name of the kernel, in ['']  # TODO

    Returns:
        kernel (Callable[np.ndarray,np.ndarray,float -> np.ndarray]): the actual kernel function, to call as ``K = kernel(X,Y,gamma)``

    .. todo::
        Write the docstring
    """
    if arg not in local_kernels_dict:
        raise NotImplementedError(f'{arg} kernel is not implemented')

    if local_kernels_dict[arg] is None:
        raise RuntimeError(f'cannot use {arg} kernel. check your installation')

    return local_kernels_dict[arg]


def get_global_kernel(arg, local_kernel):
    """

    .. todo::
        Write the docstring
    """
    gkernel, options = arg

    if gkernel not in global_kernels_dict:
        raise NotImplementedError(f'{arg} kernel is not implemented')

    return lambda x, y, s: get_global_K(x, y, s, local_kernel, global_kernels_dict[gkernel], options)


def get_kernel(arg, arg2=None):
  """ Returns the kernel function depending on the cli argument

  .. todo::
      Write the docstring
  """

  local_kernel = get_local_kernel(arg)

  if arg2 is None or arg2[0] is None:
      return local_kernel
  else:
      return get_global_kernel(arg2, local_kernel)


def train_test_split_idx(y, idx_test=None, idx_train=None,
                         test_size=defaults.test_size, random_state=defaults.random_state):
    """ Perfrom test/train data split based on random shuffling or given indices.

        If neither `idx_test` nor `idx_train` are specified, the splitting
           is done randomly using `random_state`.
        If either `idx_test` or `idx_train` is specified, the rest idx are used
           as the counterpart.
        If both `idx_test` and `idx_train` are specified, they are returned.
        * Duplicates within `idx_test` and `idx_train` are not allowed.
        * `idx_test` and `idx_train` may overlap but a warning is raised.

    Args:
        y (numpy.1darray(Nsamples)): array containing the target property of all Nsamples
        test_size (float or int): test set fraction (or number of samples)
        idx_test ([int] / numpy.1darray): list of indices for the test set (based on the sequence in X)
        idx_train ([int] / numpy.1darray): list of indices for the training set (based on the sequence in X)
        random_state (int): the seed used for random number generator (controls train/test splitting)

    Returns:
        numpy.1darray(Ntest, dtype=int) : test indices
        numpy.1darray(Ntrain, dtype=int) : train indices
        numpy.1darray(Ntest, dtype=float) : test set target property
        numpy.1darray(Ntrain, dtype=float) : train set target property
    """

    from sklearn.model_selection import train_test_split

    if idx_test is None and idx_train is None:
        idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
    elif idx_test is not None and idx_train is None:
        idx_train = np.delete(np.arange(len(y)), idx_test)
        # Check there is no repeating indices in `idx_test`.
        # Note that negative indices could be used (`np.delete` handles them)
        # this is why a check if some indices are duplicated is not sufficient.
        if len(idx_test)+len(idx_train)!=len(y):
            raise RuntimeError("Repeated test indices")
    elif idx_test is None and idx_train is not None:
        idx_test = np.delete(np.arange(len(y)), idx_train)
        if len(idx_test)+len(idx_train)!=len(y):
            raise RuntimeError("Repeated test indices")
    else:
        if len(np.delete(np.arange(len(y)), idx_train)) != len(y)-len(idx_train):
            raise RuntimeError("Repeated train indices")
        if len(np.delete(np.arange(len(y)), idx_test)) != len(y)-len(idx_test):
            raise RuntimeError("Repeated test indices")
        if len(np.delete(np.arange(len(y)), idx_test+idx_train)) != len(y)-len(idx_test)-len(idx_train):
            warnings.warn('Train and test set indices overlap. Is it intended?', RuntimeWarning, stacklevel=2)
    return np.array(idx_train), np.array(idx_test), y[idx_train], y[idx_test]


def sparse_regression_kernel(K_train, y_train, sparse_idx, eta):
    r""" Compute the sparse regression matrix and vector.

        Solution of a sparse regression problem is
        $$ \vec w = \left( \mathbf{K}_{MN} \mathbf{K}_{NM} + \eta \mathbf{1} \right) ^{-1} \mathbf{K}_{MN}\vec y $$
        where
            w: regression weights
            N: training set
            M: sparse regression set
            y: target
            K: kernel
        This function computes K_solve: $\mathbf{K}_{MN} \mathbf{K}_{NM} + \eta \mathbf{1}$
        and y_solve $\mathbf{K}_{MN}\vec y$.

    Args:
        K_train (numpy.1darray(Ntrain1,Ntrain): kernel computed on the training set.
                Ntrain1 (N in the equation) may differ from the full training set Ntrain (e.g. a subset)
        y_train (numpy.1darray(Ntrain)): array containing the target property of the full training set
        sparse_idx (numpy.1darray of int) : (M in the equation): sparse subset indices
                   wrt to the order of the full training set.
        eta (float): regularization strength for matrix inversion

    Returns:
        numpy.2darray((len(sparse), len(sparse)), dtype=float) : matrix to be inverted
        numpy.1darray((len(sparse)), dtype=float) : vector of the constant terms
    """
    K_NM    = K_train[:,sparse_idx]
    K_solve = K_NM.T @ K_NM
    K_solve[np.diag_indices_from(K_solve)] += eta
    y_solve = K_NM.T @ y_train
    return K_solve, y_solve
