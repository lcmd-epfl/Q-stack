#!/usr/bin/env python3

import numpy as np
from .kernel_utils import get_kernel, defaults, ParseKwargs, train_test_split_idx, sparse_regression_kernel
from qstack.mathutils.fps import do_fps


def condition(X, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
              akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
              test_size=defaults.test_size, idx_test=None, idx_train=None,
              sparse=None, random_state=defaults.random_state):
    """ Compute kernel matrix condition number

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        read_kernel (bool): if 'X' is a kernel and not an array of representations
        sigma (float): width of the kernel
        eta (float): regularization strength for matrix inversion
        akernel (str): local kernel (Laplacian, Gaussian, linear)
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (list): list of indices for the test set (based on the sequence in X)
        idx_train (list): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression

    Returns:
        float : condition number
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
    import argparse
    from qstack.tools import correct_num_threads
    parser = argparse.ArgumentParser(description='This program computes the condition number for the kernel matrix.')
    parser.add_argument('--x',                type=str,            dest='repr',         required=True,                 help='path to the representations file')
    parser.add_argument('--eta',              type=float,          dest='eta',          default=defaults.eta,          help='eta hyperparameter (default='+str(defaults.eta)+')')
    parser.add_argument('--sigma',            type=float,          dest='sigma',        default=defaults.sigma,        help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel',           type=str,            dest='kernel',       default=defaults.kernel,       help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel',          type=str,            dest='gkernel',      default=defaults.gkernel,      help='global kernel type (avg for average kernel, rem for REMatch kernel) (default '+str(defaults.gkernel)+')')
    parser.add_argument('--gdict', nargs='*', action=ParseKwargs,  dest='gdict',        default=defaults.gdict,        help='dictionary like input string to initialize global kernel parameters')
    parser.add_argument('--test',             type=float,          dest='test_size',    default=defaults.test_size,    help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--ll',               action='store_true', dest='ll',           default=False,                 help='if correct for the numper of threads')
    parser.add_argument('--readkernel',       action='store_true', dest='readk',        default=False,                 help='if X is kernel')
    parser.add_argument('--sparse',           type=int,            dest='sparse',       default=None,                  help='regression basis size for sparse learning')
    parser.add_argument('--random_state',     type=int,            dest='random_state', default=defaults.random_state, help='seed for the numpy.random.RandomState for test / train split generator')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    X = np.load(args.repr)
    c = condition(X, read_kernel=args.readk, sigma=args.sigma, eta=args.eta,
                  akernel=args.kernel, gkernel=args.gkernel, gdict=args.gdict,
                  test_size=args.test_size, sparse=args.sparse, random_state=args.random_state)
    print("%.1e"%c)


if __name__ == "__main__":
    main()
