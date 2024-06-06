#!/usr/bin/env python3

import numpy as np
from qstack.regression.kernel_utils import get_kernel, defaults, train_test_split_idx


def condition(X, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
              akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
              test_size=defaults.test_size, idx_test=None, idx_train=None,
              random_state=defaults.random_state):
    """

    .. todo::
        Write the docstring
    """

    idx_train, _, _, _ = train_test_split_idx(y=np.arange(len(X)), idx_test=idx_test, idx_train=idx_train,
                                              test_size=test_size, random_state=random_state)
    if read_kernel is False:
        kernel = get_kernel(akernel, [gkernel, gdict])
        X_train = X[idx_train]
        K_all  = kernel(X_train, X_train, 1.0/sigma)
    else:
        K_all  = X[np.ix_(idx_train,idx_train)]
    K_all[np.diag_indices_from(K_all)] += eta
    cond   = np.linalg.cond(K_all)
    return cond


def main():
    import argparse
    from qstack.tools import correct_num_threads
    parser = argparse.ArgumentParser(description='This program computes the condition number for the kernel matrix.')
    parser.add_argument('--x',      type=str,   dest='repr',      required=True,              help='path to the representations file')
    parser.add_argument('--eta',    type=float, dest='eta',       default=defaults.eta,       help='eta hyperparameter (default='+str(defaults.eta)+')')
    parser.add_argument('--sigma',  type=float, dest='sigma',     default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel', type=str,   dest='kernel',    default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--ll',     action='store_true', dest='ll', default=False,  help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X = np.load(args.repr)
    c = condition(X, sigma=args.sigma, eta=args.eta, akernel=args.kernel, test_size=args.test_size)
    print("%.1e"%c)


if __name__ == "__main__":
    main()
