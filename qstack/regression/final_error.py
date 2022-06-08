#!/usr/bin/env python3
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def final_error(X, y, sigma=defaults.sigma, eta=defaults.eta, akernel=defaults.kernel, test_size=defaults.test_size):
    kernel = get_kernel(akernel)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    K_all  = kernel(X_train, X_train, 1.0/sigma)
    Ks_all = kernel(X_test,  X_train, 1.0/sigma)
    K_all[np.diag_indices_from(K_all)] += eta
    alpha = scipy.linalg.solve(K_all, y_train)
    y_kf_predict = np.dot(Ks_all, alpha)
    aes = np.abs(y_test-y_kf_predict)
    return aes

def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='This program computes the full-training error for each molecule.')
    parser.add_argument('--x',      type=str,   dest='repr',      required=True,              help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',      required=True,              help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size', default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--eta',    type=float, dest='eta',       default=defaults.eta,       help='eta hyperparameter (default='+str(defaults.eta)+')')
    parser.add_argument('--sigma',  type=float, dest='sigma',     default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel', type=str,   dest='kernel',    default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--ll',     action='store_true', dest='ll', default=False,  help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    aes = final_error(X, y, sigma=args.sigma, eta=args.eta, akernel=args.kernel, test_size=args.test_size)
    np.savetxt(sys.stdout, aes, fmt='%e')

if __name__ == "__main__":
    main()

