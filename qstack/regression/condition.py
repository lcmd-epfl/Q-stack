#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def condition(X, sigma=defaults.sigma, eta=defaults.eta, akernel=defaults.kernel, test_size=defaults.test_size):
    kernel = get_kernel(akernel)
    X_train, X_test, y_train, y_test = train_test_split(X, np.arange(len(X)), test_size=test_size, random_state=0)
    K_all  = kernel(X_train, X_train, 1.0/sigma)
    K_all[np.diag_indices_from(K_all)] += eta
    cond   = np.linalg.cond(K_all)
    return cond

def main():
    import argparse
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

