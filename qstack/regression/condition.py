"""Kernel matrix condition number."""

import numpy as np
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx, sparse_regression_kernel
from .parser import RegressionParser


def condition(X, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
              akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
              test_size=defaults.test_size, idx_test=None, idx_train=None,
              sparse=None, random_state=defaults.random_state):
    """ Compute kernel matrix condition number

    Args:
        X (numpy.ndarray[Nsamples,...]): Array containing the representations of all Nsamples.
        read_kernel (bool): If 'X' is a kernel and not an array of representations.
        sigma (float): Width of the kernel.
        eta (float): Regularization strength for matrix inversion.
        akernel (str): Local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine').
        gkernel (str): Global kernel (None, 'REM', 'avg').
        gdict (dict): Parameters of the global kernels.
        test_size (float or int): Test set fraction (or number of samples).
        random_state (int): The seed used for random number generator (controls train/test splitting).
        idx_test (numpy.1darray): List of indices for the test set (based on the sequence in X).
        idx_train (numpy.1darray): List of indices for the training set (based on the sequence in X).
        sparse (int): The number of reference environnments to consider for sparse regression.

    Returns:
        float: Condition number.
    """
    idx_train, _, _, _ = train_test_split_idx(y=np.arange(len(X)), idx_test=idx_test, idx_train=idx_train,
                                              test_size=test_size, random_state=random_state)
    if read_kernel is False:
        kernel = get_kernel(akernel, [gkernel, gdict])
        X_train = X[idx_train]
        K_all  = kernel(X_train, X_train, 1.0/sigma)
    else:
        K_all  = X[np.ix_(idx_train,idx_train)]

    if not sparse:
        K_all[np.diag_indices_from(K_all)] += eta
        K_solve = K_all
    else:
        if read_kernel:
            raise RuntimeError('Cannot do FPS with kernels')
        sparse_idx = do_fps(X_train)[0][:sparse]
        K_solve, _ = sparse_regression_kernel(K_all, np.zeros(len(K_all)), sparse_idx, eta)

    cond = np.linalg.cond(K_solve)
    return cond


def main():
    """Command-line entry point for computing kernel matrix condition numbers."""
    parser = RegressionParser(description='This program computes the condition number for the kernel matrix.', hyperparameters_set='single')
    parser.remove_argument('prop')
    parser.remove_argument('train_size')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    X = np.load(args.repr)
    c = condition(X, read_kernel=args.readk, sigma=args.sigma, eta=args.eta,
                  akernel=args.kernel, gkernel=args.gkernel, gdict=args.gdict,
                  test_size=args.test_size, sparse=args.sparse, random_state=args.random_state)
    print(f"{c:.1e}")


if __name__ == "__main__":
    main()
