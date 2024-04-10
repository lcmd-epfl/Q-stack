#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def oos(X, X_oos, alpha, sigma=defaults.sigma, akernel=defaults.kernel, test_size=defaults.test_size):
    """

    .. todo::
        Write the docstring
    """
    kernel = get_kernel(akernel)
    X_train, _, _, _ = train_test_split(X, np.zeros(len(X)), test_size=test_size, random_state=0)
    K = kernel(X_oos, X_train, 1.0/sigma)
    y = K @ alpha
    return y

def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='This program makes prediction for OOS.')
    parser.add_argument('--x',      type=str,       dest='repr',      required=True,              help='path to the representations file')
    parser.add_argument('--x-oos',  type=str,       dest='x_oos',     required=True,              help='path to the OOS representations file')
    parser.add_argument('--alpha',  type=str,       dest='alpha',     required=True,              help='path to the regression weights file')
    parser.add_argument('--test',   type=float,     dest='test_size', default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--sigma',  type=float,     dest='sigma',     default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel', type=str,       dest='kernel',    default=defaults.kernel,    help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--ll',     action='store_true', dest='ll',   default=False,              help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X       = np.load(args.repr)
    X_oos   = np.load(args.x_oos)
    alpha   = np.load(args.alpha)
    y = oos(X, X_oos, alpha, sigma=args.sigma, akernel=args.kernel, test_size=args.test_size)
    np.savetxt(sys.stdout, y, fmt='%e')

if __name__ == "__main__":
    main()

