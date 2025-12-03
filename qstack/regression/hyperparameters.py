"""Hyperparameter optimization."""

import sys
import numpy as np
import scipy
from sklearn.model_selection import KFold
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx, sparse_regression_kernel
from .parser import RegressionParser


def hyperparameters(X, y,
           sigma=defaults.sigmaarr, eta=defaults.etaarr, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
           test_size=defaults.test_size, splits=defaults.splits, idx_test=None, idx_train=None,
           printlevel=0, adaptive=False, read_kernel=False, sparse=None, random_state=defaults.random_state):
    """Perform a Kfold cross-validated hyperparameter optimization (for width of kernel and regularization parameter).

    Args:
        X (numpy.ndarray[Nsamples,...]): Array containing the representations of all Nsamples.
        y (numpy.1darray[Nsamples]): Array containing the target property of all Nsamples.
        sigma (list): List of kernel width for the grid search.
        eta (list): List of regularization strength for the grid search.
        akernel (str): Local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine').
        gkernel (str): Global kernel (None, 'REM', 'avg').
        gdict (dict): Parameters of the global kernels.
        test_size (float or int): Test set fraction (or number of samples).
        splits (int): K number of splits for the Kfold cross-validation.
        idx_test (numpy.1darray): List of indices for the test set (based on the sequence in X).
        idx_train (numpy.1darray): List of indices for the training set (based on the sequence in X).
        printlevel (int): Controls level of output printing.
        adaptive (bool): To expand the grid search adaptatively.
        read_kernel (bool): If 'X' is a kernel and not an array of representations.
        sparse (int): The number of reference environnments to consider for sparse regression.
        random_state (int): The seed used for random number generator (controls train/test splitting).

    Returns:
        The results of the grid search as a numpy.2darray [Cx(MAE,std,eta,sigma)],
            where C is the number of parameter set and
            the array is sorted according to MAEs (last is minimum)

    Raises:
        RuntimeError: If 'X' is a kernel and sparse regression is chosen.
    """
    def k_fold_opt(K_all, eta):
        kfold = KFold(n_splits=splits, shuffle=False)
        all_maes = []
        for train_idx, test_idx in kfold.split(X_train):
            y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]

            if not sparse:
                K_solve = np.copy(K_all [np.ix_(train_idx,train_idx)])
                K_solve[np.diag_indices_from(K_solve)] += eta
                y_solve = y_kf_train
                Ks = K_all [np.ix_(test_idx,train_idx)]
            else:
                K_solve, y_solve = sparse_regression_kernel(K_all[train_idx], y_kf_train, sparse_idx, eta)
                Ks = K_all [np.ix_(test_idx,sparse_idx)]

            try:
                alpha = scipy.linalg.solve(K_solve, y_solve, assume_a='pos', overwrite_a=True)
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
                mean, std = k_fold_opt(K_all, e)
                if printlevel>0 :
                    sys.stderr.flush()
                    print(s, e, mean, std, flush=True)
                errors.append((mean, std, e, s))
        return errors
    if gkernel is None:
        gwrap = None
    else:
        gwrap = [gkernel, gdict]
    kernel = get_kernel(akernel, gwrap)

    idx_train, _, y_train, _ = train_test_split_idx(y=y, idx_test=idx_test, idx_train=idx_train,
                                                    test_size=test_size, random_state=random_state)
    if read_kernel is False:
        X_train = X[idx_train]
    else:
        X_train = X[np.ix_(idx_train,idx_train)]
        sigma = [np.nan]

    if sparse:
        if read_kernel:
            raise RuntimeError('Cannot do FPS with kernels')
        sparse_idx = do_fps(X_train)[0][:sparse]

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
            if   best_sigma==max(work_sigma):
                direction = 'up'
            elif best_sigma==min(work_sigma):
                direction = 'down'

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


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = RegressionParser(description='This program finds the optimal hyperparameters.', hyperparameters_set='array')
    parser.remove_argument("random_state")
    parser.remove_argument("train_size")
    return parser


def main():
    """Command-line entry point for hyperparameter optimization."""
    args = _get_arg_parser().parse_args()
    if args.readk:
        args.sigma = [np.nan]
    if args.ll:
        correct_num_threads()
    print(vars(args))

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)

    errors = hyperparameters(X, y, read_kernel=args.readk, sigma=args.sigma, eta=args.eta,
                             akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict,
                             sparse=args.sparse,
                             test_size=args.test_size, splits=args.splits, printlevel=args.printlevel, adaptive=args.adaptive)
    errors = np.array(errors)
    if args.nameout is not None:
        np.savetxt(args.nameout, errors, header="error        stdev          eta          sigma")
    print()
    print('error        stdev          eta          sigma')
    for error in errors:
        print("{:e} {:e} | {:e} {:e}".format(*tuple(error)))


if __name__ == "__main__":
    main()
