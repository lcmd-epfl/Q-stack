import sys
import numpy as np
import scipy
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx, sparse_regression_kernel
from .parser import RegressionParser


def final_error(X, y, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
                akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
                test_size=defaults.test_size, idx_test=None, idx_train=None,
                sparse=None, random_state=defaults.random_state,
                return_pred=False, return_alpha=False):
    """ Perform prediction on the test set using the full training set.

    Args:
        X (numpy.ndarray[Nsamples,...]): array containing the representations of all Nsamples
        y (numpy.1darray[Nsamples]): array containing the target property of all Nsamples
        read_kernel (bool): if 'X' is a kernel and not an array of representations
        sigma (float): width of the kernel
        eta (float): regularization strength for matrix inversion
        akernel (str): local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine')
        gkernel (str): global kernel (None, 'REM', 'avg')
        gdict (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (numpy.1darray): list of indices for the test set (based on the sequence in X)
        idx_train (numpy.1darray): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression
        return_pred (bool) : return predictions
        return_alpha (bool) : return regression weights

    Returns:
        np.1darray(Ntest) : prediction absolute errors on the test set
        np.1darray(Ntest) : (if return_pred is True) predictions on the test set
        np.1darray(Ntrain or sparse) : (if return_alpha is True) regression weights
    """
    idx_train, idx_test, y_train, y_test = train_test_split_idx(y=y, idx_test=idx_test, idx_train=idx_train,
                                                                test_size=test_size, random_state=random_state)
    if read_kernel is False:
        kernel = get_kernel(akernel, [gkernel, gdict])
        X_train, X_test = X[idx_train], X[idx_test]
        K_all  = kernel(X_train, X_train, 1.0/sigma)
        Ks_all = kernel(X_test,  X_train, 1.0/sigma)
    else:
        K_all  = X[np.ix_(idx_train,idx_train)]
        Ks_all = X[np.ix_(idx_test, idx_train)]

    if not sparse:
        K_all[np.diag_indices_from(K_all)] += eta
        K_solve, y_solve, Ks = K_all, y_train, Ks_all
    else:
        if read_kernel:
            raise RuntimeError('Cannot do FPS with kernels')
        sparse_idx = do_fps(X_train)[0][:sparse]
        K_solve, y_solve = sparse_regression_kernel(K_all, y_train, sparse_idx, eta)
        Ks = Ks_all[:,sparse_idx]

    alpha = scipy.linalg.solve(K_solve, y_solve, assume_a='pos')
    y_kf_predict = np.dot(Ks, alpha)
    aes = np.abs(y_test-y_kf_predict)
    if return_pred and return_alpha:
        return aes, y_kf_predict, alpha
    elif return_pred:
        return aes, y_kf_predict
    elif return_alpha:
        return aes, alpha
    else:
        return aes


def main():
    """Command-line entry point for computing final prediction errors."""
    parser = RegressionParser(description='This program computes the full-training error for each molecule.', hyperparameters_set='single')
    parser.remove_argument('train_size')
    parser.add_argument('--save-alpha', type=str,   dest='save_alpha',  default=None,  help='file to write the regression coefficients to')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    aes, _pred, alpha = final_error(X, y, read_kernel=args.readk, sigma=args.sigma, eta=args.eta,
                                    akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict,
                                    test_size=args.test_size, sparse=args.sparse, random_state=args.random_state,
                                    return_pred=True, return_alpha=True)
    if args.save_alpha:
        np.save(args.save_alpha, alpha)
    np.savetxt(sys.stdout, aes, fmt='%e')


if __name__ == "__main__":
    main()
