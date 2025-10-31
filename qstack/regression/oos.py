import sys
import numpy as np
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx
from .parser import RegressionParser


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
    """Command-line entry point for out-of-sample predictions."""
    parser = RegressionParser(description='This program makes prediction for OOS.', hyperparameters_set='single')
    parser.remove_argument('prop')
    parser.remove_argument('train_size')
    parser.remove_argument('readk')
    parser.remove_argument('eta')
    parser.add_argument('--x-oos',  type=str, dest='x_oos',  required=True,  help='path to the OOS representations file')
    parser.add_argument('--alpha',  type=str, dest='alpha',  required=True,  help='path to the regression weights file')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    X       = np.load(args.repr)
    X_oos   = np.load(args.x_oos)
    alpha   = np.load(args.alpha)
    y = oos(X, X_oos, alpha, sigma=args.sigma,
            akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict,
            test_size=args.test_size, sparse=args.sparse, random_state=args.random_state)
    np.savetxt(sys.stdout, y, fmt='%e')


if __name__ == "__main__":
    main()
