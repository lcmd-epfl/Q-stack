#!/usr/bin/env python3

import numpy as np
import scipy
from qstack.regression.kernel_utils import get_kernel, defaults, train_test_split_idx


def final_error(X, y, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
                akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
                test_size=defaults.test_size, idx_test=None, idx_train=None,
                random_state=defaults.random_state,
                return_pred=False, return_alpha=False, save_alpha=None):
    """

    .. todo::
        Write the docstring
    """
    idx_train, idx_test, y_train, y_test = train_test_split_idx(y=y, idx_test=idx_test, idx_train=idx_train,
                                                                test_size=test_size, random_state=random_state)
    if read_kernel is False:
        kernel = get_kernel(akernel, [gkernel, gdict])
        X_train, X_test = X[idx_train], X[idx_test]
        K_all  = kernel(X_train, X_train, 1.0/sigma)
        Ks_all = kernel(X_test,  X_train, 1.0/sigma)
    else:
        K_all  = X[np.ix_(idx_train,idx_train)]
        Ks_all = X[np.ix_(idx_test, idx_train)]

    K_all[np.diag_indices_from(K_all)] += eta
    alpha = scipy.linalg.solve(K_all, y_train, assume_a='pos')
    y_kf_predict = np.dot(Ks_all, alpha)
    aes = np.abs(y_test-y_kf_predict)
    if save_alpha:
        np.save(save_alpha, alpha)
    if return_pred and return_alpha:
        return aes, y_kf_predict, alpha
    elif return_pred:
        return aes, y_kf_predict
    elif return_alpha:
        return aes, alpha
    else:
        return aes


def main():
    import sys
    import argparse
    from qstack.tools import correct_num_threads
    parser = argparse.ArgumentParser(description='This program computes the full-training error for each molecule.')
    parser.add_argument('--x',          type=str,   dest='repr',        required=True,              help='path to the representations file')
    parser.add_argument('--y',          type=str,   dest='prop',        required=True,              help='path to the properties file')
    parser.add_argument('--test',       type=float, dest='test_size',   default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--eta',        type=float, dest='eta',         default=defaults.eta,       help='eta hyperparameter (default='+str(defaults.eta)+')')
    parser.add_argument('--sigma',      type=float, dest='sigma',       default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel',     type=str,   dest='kernel',      default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--save-alpha', type=str,   dest='save_alpha',  default=None,               help='file to write the regression coefficients to (default None)')
    parser.add_argument('--ll',     action='store_true', dest='ll',     default=False,              help='if correct for the numper of threads')
    parser.add_argument('--random_state',  type=int, dest='random_state', default=defaults.random_state,  help='random state for test / train splitting')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    aes = final_error(X, y, sigma=args.sigma, eta=args.eta, akernel=args.kernel, test_size=args.test_size, save_alpha=args.save_alpha, random_state=random_state)
    np.savetxt(sys.stdout, aes, fmt='%e')


if __name__ == "__main__":
    main()
