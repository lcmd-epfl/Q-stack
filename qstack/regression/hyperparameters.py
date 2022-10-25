#!/usr/bin/env python3

import sys
import numpy as np
import scipy
from sklearn.model_selection import train_test_split, KFold
from qstack.regression.kernel_utils import get_kernel, defaults, ParseKwargs
from qstack.tools import correct_num_threads

def hyperparameters(X, y,
           sigma=defaults.sigmaarr, eta=defaults.etaarr, gkernel=defaults.gkernel, gdict=defaults.gdict,
           akernel=defaults.kernel, test_size=defaults.test_size, splits=defaults.splits,
           printlevel=0, adaptive=False, read_kernel=False):

    def k_fold_opt(K_all):
        kfold = KFold(n_splits=splits, shuffle=False)
        all_maes = []
        for train_idx, test_idx in kfold.split(X_train):
            y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]
            K  = K_all [np.ix_(train_idx,train_idx)]
            Ks = K_all [np.ix_(test_idx,train_idx)]
            try:
                alpha = scipy.linalg.solve(K, y_kf_train, assume_a='pos', overwrite_a=True)
            except scipy.linalg.LinAlgError:
                print('singular matrix')
                all_maes.append(np.nan)
                break
            y_kf_predict = np.dot(Ks, alpha)
            all_maes.append(np.mean(np.abs(y_kf_predict-y_kf_test)))
        return np.mean(all_maes), np.std(all_maes)

    def hyper_loop(sigma, eta):
        errors = []
        for s in sigma:
            if read_kernel is False:
                K_all = kernel(X_train, X_train, 1.0/s)
            else:
                K_all = X_train

            for e in eta:
                K_all[np.diag_indices_from(K_all)] += e
                mean, std = k_fold_opt(K_all)
                K_all[np.diag_indices_from(K_all)] -= e
                if printlevel>0 :
                    sys.stderr.flush()
                    print(s, e, mean, std, flush=True)
                errors.append((mean, std, e, s))
        return errors

    kernel = get_kernel(akernel, [gkernel, gdict])
    if read_kernel is False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    else:
        idx_train, idx_test, y_train, y_test = train_test_split(np.arange(len(y)), y, test_size=test_size, random_state=0)
        X_train = X[np.ix_(idx_train,idx_train)]
        sigma = [np.nan]

    work_sigma = sigma
    errors = []
    direction = None
    while True:
        errors = list(errors)
        errors.extend(hyper_loop(work_sigma, eta))
        errors = np.array(errors)
        ind = np.argsort(errors[:,0])[::-1]
        errors = errors[ind]

        if not adaptive:
            break

        best_sigma = errors[-1][3]
        new_sigma = None

        if direction is None:
            if   best_sigma==max(work_sigma): direction = 'up'
            elif best_sigma==min(work_sigma): direction = 'down'

        # at the 1st iteration if is checked twice on purpose
        if direction=='up'     and best_sigma==max(work_sigma):
                new_sigma = best_sigma*np.array(defaults.sigmaarr_mult[1:])
        elif direction=='down' and best_sigma==min(work_sigma):
                new_sigma = best_sigma/np.array(defaults.sigmaarr_mult[1:])

        if new_sigma is None:
            break
        work_sigma = new_sigma
        print('next iteration:', work_sigma, flush=True)
    return errors

def main():
    import argparse
    parser = argparse.ArgumentParser(description='This program finds the optimal hyperparameters.')
    parser.add_argument('--x',      type=str,   dest='repr',       required=True, help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',       required=True, help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--akernel',     type=str,   dest='akernel',     default=defaults.kernel,    help='local kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel',     type=str,   dest='gkernel',     default=defaults.gkernel,    help='global kernel type (avg for average kernel, rem for REMatch kernel) (default '+defaults.gkernel+')')
    parser.add_argument('--gdict',     nargs='*',   action=ParseKwargs, dest='gdict',     default=defaults.gdict,    help='dictionary like input string to initialize global kernel parameters')
    parser.add_argument('--splits', type=int,   dest='splits',     default=defaults.splits,    help='k in k-fold cross validation (default='+str(defaults.n_rep)+')')
    parser.add_argument('--print',  type=int,   dest='printlevel', default=0,                  help='printlevel')
    parser.add_argument('--eta',    type=float, dest='eta',   default=defaults.etaarr,   nargs='+', help='eta array')
    parser.add_argument('--sigma',  type=float, dest='sigma', default=defaults.sigmaarr, nargs='+', help='sigma array')
    parser.add_argument('--ll',   action='store_true', dest='ll',       default=False,  help='if correct for the numper of threads')
    parser.add_argument('--ada',  action='store_true', dest='adaptive', default=False,  help='if adapt sigma')
    parser.add_argument('--readkernel', action='store_true', dest='readk', default=False,  help='if X is kernel')
    args = parser.parse_args()
    if(args.readk): args.sigma = [np.nan]
    print(vars(args))
    if(args.ll): correct_num_threads()

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    errors = hyperparameters(X, y, read_kernel=args.readk, sigma=args.sigma, eta=args.eta, akernel=args.kernel, test_size=args.test_size, splits=args.splits, printlevel=args.printlevel, adaptive=args.adaptive)

    print()
    print('error        stdev          eta          sigma')
    for error in errors:
        print("%e %e | %e %e" % tuple(error))

if __name__ == "__main__":
    main()
