#!/usr/bin/env python3

import numpy as np
import scipy
from sklearn.model_selection import train_test_split, KFold
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def hyperparameters(X, y,
           sigma=defaults.sigmaarr, eta=defaults.etaarr,
           akernel=defaults.kernel, test_size=defaults.test_size, splits=defaults.splits,
           printlevel=0):

    def k_fold_opt(eta, sigma, printlevel):
        K_all = kernel(X_train, X_train, 1.0/sigma)
        K_all[np.diag_indices_from(K_all)] += eta
        kfold = KFold(n_splits=splits, shuffle=False)
        all_maes = []
        for train_idx, test_idx in kfold.split(X_train):
            y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]
            K  = K_all [np.ix_(train_idx,train_idx)]
            Ks = K_all [np.ix_(test_idx,train_idx)]
            alpha = scipy.linalg.solve(K, y_kf_train, assume_a='pos')
            y_kf_predict = np.dot(Ks, alpha)
            all_maes.append(np.mean(np.abs(y_kf_predict-y_kf_test)))
        mean = np.mean(all_maes)
        std  = np.std(all_maes)
        if printlevel>0 :print(sigma, eta, mean, std, flush=True)
        return mean, std, eta, sigma

    kernel = get_kernel(akernel)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    errors = []
    for e in eta:
        for s in sigma:
            errors.append(k_fold_opt(e, s, printlevel))
    errors = np.array(errors)
    ind = np.argsort(errors[:,0])[::-1]
    errors = errors[ind]
    return errors

def main():
    import argparse
    parser = argparse.ArgumentParser(description='This program finds the optimal hyperparameters.')
    parser.add_argument('--x',      type=str,   dest='repr',       required=True, help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',       required=True, help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--kernel', type=str,   dest='kernel',     default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--splits', type=int,   dest='splits',     default=defaults.splits,    help='k in k-fold cross validation (default='+str(defaults.n_rep)+')')
    parser.add_argument('--print',  type=int,   dest='printlevel', default=0,                  help='printlevel')
    parser.add_argument('--eta',    type=float, dest='eta',   default=defaults.etaarr,   nargs='+', help='eta array')
    parser.add_argument('--sigma',  type=float, dest='sigma', default=defaults.sigmaarr, nargs='+', help='sigma array')
    parser.add_argument('--ll',   action='store_true', dest='ll', default=False,  help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    errors = hyperparameters(X, y, sigma=args.sigma, eta=args.eta, akernel=args.kernel, test_size=args.test_size, splits=args.splits, printlevel=args.printlevel)

    print()
    print('error        stdev          eta          sigma')
    for error in errors:
        print("%e %e | %e %f" % tuple(error))

if __name__ == "__main__":
    main()
