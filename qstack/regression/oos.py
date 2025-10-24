#!/usr/bin/env python3

import numpy as np
from qstack.regression.kernel_utils import get_kernel, defaults, ParseKwargs, train_test_split_idx
from qstack.mathutils.fps import do_fps


def oos(X, X_oos, alpha, sigma=defaults.sigma,
        akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
        test_size=defaults.test_size, idx_test=None, idx_train=None,
        sparse=None, random_state=defaults.random_state):
    """ Perform prediction on an out-of-sample (OOS) set.

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        X_oos (numpy.2darray[Noos,Nfeat]): array of OOS representations.
        alpha (numpy.1darray(Ntrain or sparse)): regression weights.
        sigma (float): width of the kernel
        akernel (str): local kernel (Laplacian, Gaussian, linear)
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (list): list of indices for the test set (based on the sequence in X)
        idx_train (list): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression

    Returns:
        np.1darray(Noos) : predictions on the OOS set
    """

    idx_train, _, _, _, = train_test_split_idx(y=np.arange(len(X)), idx_test=idx_test, idx_train=idx_train,
                                               test_size=test_size, random_state=random_state)
    kernel = get_kernel(akernel, [gkernel, gdict])
    X_train = X[idx_train]
    if sparse:
        sparse_idx = do_fps(X_train)[0][:sparse]
        X_train = X_train[sparse_idx]
    K = kernel(X_oos, X_train, 1.0/sigma)
    y = K @ alpha
    return y


def main():
    import sys
    import argparse
    from qstack.tools import correct_num_threads
    parser = argparse.ArgumentParser(description='This program makes prediction for OOS.')
    parser.add_argument('--x',      type=str,       dest='repr',      required=True,              help='path to the representations file')
    parser.add_argument('--x-oos',  type=str,       dest='x_oos',     required=True,              help='path to the OOS representations file')
    parser.add_argument('--alpha',  type=str,       dest='alpha',     required=True,              help='path to the regression weights file')
    parser.add_argument('--test',   type=float,     dest='test_size', default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--sigma',  type=float,     dest='sigma',     default=defaults.sigma,     help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--akernel',       type=str,   dest='akernel',     default=defaults.kernel,
        help='local kernel type: "G" for Gaussian, "L" for Laplacian, "dot" for dot products, "cosine" for cosine similarity, "G_sklearn","L_sklearn","G_customc","L_customc","L_custompy" for specific implementations. '
             '("L_custompy" is suited to open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel',       type=str,   dest='gkernel',     default=defaults.gkernel,    help='global kernel type (avg for average kernel, rem for REMatch kernel) (default '+str(defaults.gkernel)+')')
    parser.add_argument('--gdict',         nargs='*',   action=ParseKwargs, dest='gdict',     default=defaults.gdict,    help='dictionary like input string to initialize global kernel parameters')
    parser.add_argument('--ll',     action='store_true', dest='ll',   default=False,              help='if correct for the numper of threads')
    parser.add_argument('--sparse',           type=int,            dest='sparse',       default=None,                  help='regression basis size for sparse learning')
    parser.add_argument('--random_state',  type=int, dest='random_state', default=defaults.random_state,  help='random state for test / train splitting')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X       = np.load(args.repr)
    X_oos   = np.load(args.x_oos)
    alpha   = np.load(args.alpha)
    y = oos(X, X_oos, alpha, sigma=args.sigma,
            akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict,
            test_size=args.test_size, sparse=args.sparse, random_state=args.random_state)
    np.savetxt(sys.stdout, y, fmt='%e')


if __name__ == "__main__":
    main()
