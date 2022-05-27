#!/usr/bin/env python3
import numpy as np
from qstack.regression.kernel_utils import get_kernel, defaults

def kernel(X, sigma=defaults.sigma, akernel=defaults.kernel):
    kernel = get_kernel(akernel)
    K = kernel(X, X, 1.0/sigma)
    return K

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='This program computes kernel.')
    parser.add_argument('--x',      type=str,   dest='repr',      required=True,           help='path to the representations file')
    parser.add_argument('--sigma',  type=float, dest='sigma',     default=defaults.sigma,  help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--kernel', type=str,   dest='kernel',    default=defaults.kernel, help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--dir',    type=str,   dest='dir',       default='./',            help='directory to save the output in (default=current dir)')
    args = parser.parse_args()
    print(vars(args))
    X = np.load(args.repr)
    K = kernel(X, args.sigma, args.kernel)
    np.save(args.dir+'/K_'+os.path.basename(args.repr)+'_'+args.kernel, K)

if __name__ == "__main__":
    main()
