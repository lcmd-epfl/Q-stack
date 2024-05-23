#!/usr/bin/env python3

import sys
import numpy as np
import scipy
from sklearn.model_selection import train_test_split, KFold
from qstack.regression.kernel_utils import get_kernel, defaults, ParseKwargs
from qstack.regression.hyperparameters import hyperparameters
from qstack.regression.regression import regression
from qstack.tools import correct_num_threads


def cv(X, y,
           sigma=defaults.sigmaarr, eta=defaults.etaarr, gkernel=defaults.gkernel, gdict=defaults.gdict,
           akernel=defaults.kernel, test_size=defaults.test_size, splits=defaults.splits,
           printlevel=0, adaptive=False, read_kernel=False, ipywidget=None, n_rep=defaults.n_rep):
    hyper_runs = []
    lc_runs = []
    seeds = [123, 1, 2, 66, 666, 18, 9, 1996, 26,  3, 17]
    for seed,n in zip(seeds, range(n_rep)):
        error = hyperparameters(X, y, read_kernel=False, sigma=sigma, eta=eta, akernel=akernel, test_size=test_size, splits=splits, printlevel=printlevel, adaptive=adaptive, debug=seed)
        print(error)
        exit()
        hyper_runs.append(zip([n]*len(error), error, stdev, eta, sigma))
        maes_all = regression(X, y, read_kernel=False, sigma=sigma, eta=eta, akernel=akernel, test_size=test_size, train_size=train_size, n_rep=splits, debug=seed)
        lc_runs.append(maes_all)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='This program runs a full cross-validation of the learning curves (hyperparameters search inbcluded).')
    parser.add_argument('--x',      type=str,   dest='repr',       required=True, help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',       required=True, help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--akernel',     type=str,   dest='akernel',     default=defaults.kernel,    help='local kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel',     type=str,   dest='gkernel',     default=defaults.gkernel,    help='global kernel type (avg for average kernel, rem for REMatch kernel) (default )')
    parser.add_argument('--gdict',     nargs='*',   action=ParseKwargs, dest='gdict',     default=defaults.gdict,    help='dictionary like input string to initialize global kernel parameters')
    parser.add_argument('--splits', type=int,   dest='splits',     default=defaults.splits,    help='k in k-fold cross validation (default='+str(defaults.n_rep)+')')
    parser.add_argument('--print',  type=int,   dest='printlevel', default=0,                  help='printlevel')
    parser.add_argument('--eta',    type=float, dest='eta',   default=defaults.etaarr,   nargs='+', help='eta array')
    parser.add_argument('--sigma',  type=float, dest='sigma', default=defaults.sigmaarr, nargs='+', help='sigma array')
    parser.add_argument('--ll',   action='store_true', dest='ll',       default=False,  help='if correct for the numper of threads')
    parser.add_argument('--ada',  action='store_true', dest='adaptive', default=False,  help='if adapt sigma')
    parser.add_argument('--readkernel', action='store_true', dest='readk', default=False,  help='if X is kernel')
    parser.add_argument('--name',      type=str,   dest='nameout',       required=True, help='the name of the output file')
    parser.add_argument('--select',      type=str,   dest='f_select',       required=False, help='a txt file containing the indices of the selected representations')
    parser.add_argument('--sub',      action="store_true",   dest='sub_test',       required=False, help='run fast test (10 sub-data points)')
    args = parser.parse_args()
    if(args.readk): args.sigma = [np.nan]
    print(vars(args))
    if(args.ll): correct_num_threads()

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    if args.sub_test:
        X = X[:100]
        y = y[:100]
    if args.f_select != None:
        selected = np.loadtxt(args.f_select, dtype=int)
        X = X[selected]
        y = y[selected]
    final = cv(X, y, sigma=args.sigma, eta=args.eta, akernel=args.akernel, test_size=args.test_size, splits=args.splits, printlevel=args.printlevel, adaptive=args.adaptive)

if __name__ == '__main__' : main()
