#!/usr/bin/env python3

import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def regression(X, y, sigma=defaults.sigma, eta=defaults.eta, akernel=defaults.kernel, test_size=defaults.test_size, train_size=defaults.train_size, n_rep=defaults.n_rep, debug=False):
    kernel = get_kernel(akernel)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    all_indices_train = np.arange(X_train.shape[0])
    K_all  = kernel(X_train, X_train, 1.0/sigma)
    Ks_all = kernel(X_test,  X_train, 1.0/sigma)
    K_all[np.diag_indices_from(K_all)] += eta

    if debug:
        np.random.seed(666)

    maes_all = []
    for size in train_size:
        size_train = int(np.floor(X_train.shape[0]*size))
        maes = []
        for rep in range(n_rep):
            train_idx = np.random.choice(all_indices_train, size = size_train, replace=False)
            y_kf_train = y_train[train_idx]
            K  = K_all [np.ix_(train_idx,train_idx)]
            Ks = Ks_all[:,train_idx]
            alpha = scipy.linalg.solve(K, y_kf_train, assume_a='pos')
            y_kf_predict = np.dot(Ks, alpha)
            maes.append(np.mean(np.abs(y_test-y_kf_predict)))
        maes_all.append((size_train, np.mean(maes), np.std(maes)))
    return maes_all

def main():
    import argparse
    parser = argparse.ArgumentParser(description='This program computes the learning curve.')
    parser.add_argument('--x',      type=str,   dest='repr',       required=True, help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',       required=True, help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--eta',    type=float, dest='eta',        default=defaults.eta,       help='eta hyperparameter (default='+str(defaults.eta)+')')
    parser.add_argument('--sigma',  type=float, dest='sigma',      default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel', type=str,   dest='kernel',     default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--splits', type=int,   dest='splits',     default=defaults.n_rep,     help='number of splits (default='+str(defaults.n_rep)+')')
    parser.add_argument('--train',  type=float, dest='train_size', default=defaults.train_size, nargs='+', help='training set fractions')
    parser.add_argument('--debug',  action='store_true', dest='debug', default=False, help='enable debug')
    parser.add_argument('--ll',     action='store_true', dest='ll', default=False,  help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    maes_all = regression(X, y, sigma=args.sigma, eta=args.eta, akernel=args.kernel,
                          test_size=args.test_size, train_size=args.train_size, n_rep=args.splits, debug=args.debug)
    for size_train, meanerr, stderr in maes_all:
        print("%d\t%e\t%e" % (size_train, meanerr, stderr))

if __name__ == "__main__":
    main()

