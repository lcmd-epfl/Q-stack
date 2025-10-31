import numpy as np
import scipy
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx, sparse_regression_kernel
from .parser import RegressionParser


def regression(X, y, read_kernel=False, sigma=defaults.sigma, eta=defaults.eta,
               akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict,
               test_size=defaults.test_size, train_size=defaults.train_size, n_rep=defaults.n_rep,
               random_state=defaults.random_state, idx_test=None, idx_train=None,
               sparse=None, debug=False, save_pred=False):
    """ Produces learning curves (LC) data, for various training sizes, using kernel ridge regression and the user specified parameters

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        y (numpy.1darray[Nsamples]): array containing the target property of all Nsamples
        read_kernel (bool): if 'X' is a kernel and not an array of representations
        sigma (float): width of the kernel
        eta (float): regularization strength for matrix inversion
        akernel (str): local kernel (Laplacian, Gaussian, linear)
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        train_size (list): list of training set size fractions used to evaluate the points on the LC
        n_rep (int): the number of repetition for each point (using random sampling)
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (list): list of indices for the test set (based on the sequence in X)
        idx_train (list): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression
        debug (bool): to use a fixed seed for random sampling (for reproducibility)
        save_pred (bool): to return all predicted targets

    Returns:
        The computed LC, as a list containing all its points (train size, MAE, std)
        If save_pres is True, a tuple with (results, (target values, predicted values))
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

    all_indices_train = np.arange(len(y_train))
    if not sparse:
        K_all[np.diag_indices_from(K_all)] += eta
    else:
        if read_kernel:
            raise RuntimeError('Cannot do FPS with kernels')
        sparse_idx = do_fps(X_train)[0][:sparse] # indices within the training set

    if debug:
        # Ensures reproducibility of the sample selection for each train_size over repetitions (n_rep)
        rng = np.random.RandomState(666)
    else:
        rng = np.random.RandomState()

    maes_all = []
    for size in train_size:
        size_train = int(np.floor(len(y_train)*size)) if size <= 1.0 else size
        maes = []
        for _rep in range(n_rep):
            train_idx = rng.choice(all_indices_train, size = size_train, replace=False)
            y_kf_train = y_train[train_idx]

            if not sparse:
                K_solve = K_all [np.ix_(train_idx,train_idx)]
                y_solve = y_kf_train
                Ks = Ks_all[:,train_idx]
            else:
                K_solve, y_solve = sparse_regression_kernel(K_all[train_idx], y_kf_train, sparse_idx, eta)
                Ks = Ks_all[:,sparse_idx]

            alpha = scipy.linalg.solve(K_solve, y_solve, assume_a='pos')
            y_kf_predict = np.dot(Ks, alpha)
            maes.append(np.mean(np.abs(y_test-y_kf_predict)))
        maes_all.append((size_train, np.mean(maes), np.std(maes)))
    return maes_all if not save_pred else (maes_all, (y_test, y_kf_predict))


def main():
    """Command-line entry point for computing learning curves."""
    parser = RegressionParser(description='This program computes the learning curve.', hyperparameters_set='single')
    parser.add_argument('--splits',  type=int,            dest='splits',    default=defaults.n_rep, help='number of splits')
    parser.add_argument('--name',    type=str,            dest='nameout',   default=None,           help='the name of the output file containting the LC data (.txt)')
    parser.add_argument('--debug',   action='store_true', dest='debug',     default=False,          help='enable debug')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll):
        correct_num_threads()
    X = np.load(args.repr)
    y = np.loadtxt(args.prop)

    maes_all = regression(X, y, read_kernel=args.readk, sigma=args.sigma, eta=args.eta,
                          akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict,
                          test_size=args.test_size, train_size=args.train_size, n_rep=args.splits, sparse=args.sparse,
                          debug=args.debug, random_state=args.random_state)
    for size_train, meanerr, stderr in maes_all:
        print(f"{size_train}\t{meanerr:e}\t{stderr:e}")
    maes_all = np.array(maes_all)
    if args.nameout is not None:
        np.savetxt(args.nameout, maes_all, header="size_train, meanerr, stderr")


if __name__ == "__main__":
    main()
