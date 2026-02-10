"""Hyperparameter optimization."""

import sys
import logging
import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.utils.parallel import Parallel, delayed
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx, sparse_regression_kernel
from .parser import RegressionParser

logger = logging.getLogger("qstack.regression.hyperparameters")


# #####################
# parabola-based line search

def fit_quadratic(x1,x2,x3, y1,y2,y3):
    """Compute the three coefficients of a quadratic polynomial going through three given points.

    Could probably be replaced by `np.polyfit` now that I know about it. Fluff it, we ball.
    """
    # we need to change coordinates around for this

    # first, slopes at 0.5(x1+x2) and 0.5(x2+x3)
    # the 'meta-slope' allows us to get 2*curvature
    slope1 = (y2-y1)/(x2-x1)
    slope2 = (y3-y2)/(x3-x2)
    curv = (slope2-slope1)/(x3-x1)  # the "0.5*" disappears

    # then, remove the curvature from points 1 and 2 to determine 1st-degree term
    y1_b = y1 - curv*x1**2
    y2_b = y2 - curv*x2**2
    slope = (y2_b-y1_b)/(x2-x1)

    # finally, the intercept
    intercept = y1_b - slope*x1
    return curv, slope, intercept

def parabolic_search(x_left, x_right, get_err, n_iter=10, x_thres=0.1, y_thres=0.01):
    """Gradient-less line search of the minimum of `get_err`, supposedly between `x_left` and `x_right`.

    Fits quadratic polynomials to perform this search, meaning `get_err` is assumed to be convex.

    Args:
        x_left (float): supposed left bound of the minimum of `get_err`
        x_right (float): supposed right bound of the minimum of `get_err`
        get_err (callable float->float): the function to minimise.
        n_iter (int): the number of function calls allowed
        x_thres (float): the acceptable error threshold for the the argmin to find
        y_thres (float): the acceptable error threshold for the min to find

    Returns:
        the (argmin, min) tuple characterising the minimum of the function (2x float)
    """
    y_left = get_err(x_left)
    y_right = get_err(x_right)
    x_center = 0.5*(x_left+x_right)
    y_center = get_err(x_center)

    all_errs = [(x_left,y_left),(x_center,y_center),(x_right,y_right)]

    while y_left < y_center or y_right < y_center:
        # while it looks like we need to look elsewhere than in our original bracket
        # (because the center isn't closer to the local minimum than the bounds)
        if y_left < y_right:
            logger.debug('no local minimum in sight, extending to the left...')
            x_right, y_right = x_center, y_center
            x_center, y_center = x_left, y_left
            x_left = 2*x_center - x_right
            y_left = get_err(x_left)
            all_errs.insert(0, (x_left,y_left))
        else:
            logger.debug('no local minimum in sight, extending to the right...')
            x_left, y_left = x_center, y_center
            x_center, y_center = x_right, y_right
            x_right = 2*x_center - x_left
            y_right = get_err(x_right)
            all_errs.append((x_right,y_right))
        n_iter -= 1
        if n_iter <=0:
            break
    # after this point, either we are exiting early or we have found the right bounds
    all_errs.sort()
    logger.debug('local minimum in bounds, proceeding with parabolic search (bounds at: %r)', all_errs)
    logger.debug('chosen: %f\\%f/%f', x_left, x_center, x_right)
    while n_iter > 0:
        a,b,c = fit_quadratic(x_left, x_center, x_right, y_left, y_center, y_right)
        if a<=0:  # lol no local minimum
            logger.debug('no local minimum...')
            if x_left < x_right:
                x_new = 0.5*(x_left+x_center)
                ypred_new = np.nan
            else:
                x_new = 0.5*(x_right+x_center)
                ypred_new = np.nan
        else:
            x_new = -0.5*b/a
            ypred_new = -0.25*b**2/a + c
        y_new = get_err(x_new)
        n_iter -=1
        logger.debug('from chosen points %f\\%f/%f', x_left, x_center, x_right)
        logger.debug('predicted local minimum at %f->%f, true error %f', x_new, ypred_new, y_new)
        all_errs.append((x_new, y_new))
        all_errs.sort()
        logger.debug('current data: %r', all_errs)

        if x_new < x_left or x_new > x_right:
            logger.debug('predicted local minimum not in immediate bounds, regaining bearings...')
            new_index = np.argmin(np.asarray(all_errs)[:,1])
            if new_index in (0, len(all_errs)-1):
                raise AssertionError('edges of the search are somehow the minimum in second phase of function')
            x_left, y_left = all_errs[new_index-1]
            x_right, y_right = all_errs[new_index+1]
            x_center, y_center = all_errs[new_index]

        elif max(y_right,y_left, y_new)-min(y_new, y_center) < y_thres:
            break
        elif y_new > y_center:
            if x_new > x_center:
                x_right, y_right = x_new, y_new
            else:
                x_left, y_left = x_new, y_new
        else:  # if y_new <= y_center
            if x_new > x_center:
                x_left, y_left = x_center, y_center
                x_center, y_center = x_new, y_new
            else:
                x_right, y_right = x_center, y_center
                x_center, y_center = x_new, y_new

        if abs(x_right - x_left) < x_thres:
            break

    opt_idx = np.argmin(np.asarray(all_errs)[:,1])
    return all_errs[opt_idx]


def standard_grid_search(x_list, get_err):
    """Module-internal function: single-parameter optimisation function, using a simple grid search.

    Args:
        x_list (iterable[float]): pre-defined grid containing the values to try for the parameter
        get_err (callable float->float): process to optimise, returning the error associated with a single parameter value.

    Returns:
        val (float): optimal parameter value
        err (float): the associated error
    """
    errors = np.array(get_err(x) for x in x_list)
    xi = errors.argmin()
    return xi, errors[xi]

def adaptative_grid_search(x_list, get_err):
    """Module-internal function: single-parameter optimisation function, using an adaptive grid search.

    Operates like a standard grid search, but extends the grid if the optimal parameter value is at one of the edges of the provided grid.

    Args:
        x_list (iterable[float]): pre-defined original grid of the parameter
        get_err (callable float->float): process to optimise, returning the error associated with a single parameter value.

    Returns:
        errors (np.ndarray[N_evals,2]): list of (parameter value, error) pairs, sorted to have decreasing errors
    """
    work_list = x_list
    errors = []
    direction = None
    while True:
        errors = list(errors)
        for x in work_list:
            errors.append((x,get_err(x)))
        errors = np.array(errors)
        ind = np.argsort(errors[:,1])[::-1]
        errors = errors[ind]

        current_argmin = errors[-1,0]

        if direction is None:
            if   current_argmin==max(work_list):
                direction = 'up'
            elif current_argmin==min(work_list):
                direction = 'down'

        # at the 1st iteration if is checked twice on purpose
        if direction=='up'     and current_argmin==max(work_list):
            work_list = current_argmin*np.array(defaults.sigmaarr_mult[1:])
        elif direction=='down' and current_argmin==min(work_list):
            work_list = current_argmin/np.array(defaults.sigmaarr_mult[1:])
        else:
            break

        print('next iteration:', work_list, flush=True)
    return errors

# #####################
# main functions of the hyperparameter optimisation


def kfold_alpha_eval(K_all, y_train, n_splits, eta_list, sparse, parallel = None):
    """Module-internal function: optimise alpha (regularisation parameter) of a KRR learning model, using a K-fold validation.

    Args:
        K_all: matrix of kernel values (can be n_total*n_total for naive KRR or n_total*n_references for sparse KRR)
        y_train: learnable properties for all inputs (n_total-length vector)
        n_splits: number of folds for k-fold validation
        eta_list: all the values of eta (KRR regularisation parameter) to try (array-like)
        sparse: whether the KRR to run is sparse (bool)
        parallel: optional joblib.Parallel instance to use to parallelise this function (by default one is constructed)

    Returns:
        errors: array of "entries", each with the three values (np.ndarray, shape (len(eta_list),3) ):
            mean: mean (over k-folds) validation error for this value of eta
            stddev: standard deviation of the same error
            eta: the corresponding value of eta
    """
    if parallel is None:
        parallel = Parallel(n_jobs=-1, return_as="generator_unordered")
    kfold = KFold(n_splits=n_splits, shuffle=False)
    maes = np.full((n_splits, len(eta_list)), np.inf)
    y_train = np.asarray(y_train)


    def inner_call(fold_i, eta_i, K_all, sparse, y_train, eta, train_idx, test_idx):
        y_kf_train, y_kf_test = y_train[train_idx], y_train[test_idx]

        if not sparse:
            K_solve = K_all [np.ix_(train_idx,train_idx)]
            if np.may_share_memory(K_solve, K_all):
                K_solve = K_solve.copy()
            K_solve[np.diag_indices_from(K_solve)] += eta
            y_solve = y_kf_train
            Ks = K_all [np.ix_(test_idx,train_idx)]
        else:
            K_solve, y_solve = sparse_regression_kernel(K_all[train_idx], y_kf_train, slice(None), eta)
            Ks = K_all[test_idx]

        try:
            alpha = scipy.linalg.solve(K_solve, y_solve, assume_a='pos', overwrite_a=True)
        except scipy.linalg.LinAlgError:
            print('singular matrix')
            raise
        y_kf_predict = np.dot(Ks, alpha)
        return fold_i, eta_i, np.mean(np.abs(y_kf_predict-y_kf_test))

    mae_generator = parallel(
        delayed(inner_call)(fold_i, eta_i, K_all, sparse, y_train, eta, t,v)
        for eta_i,eta in enumerate(eta_list)
        for fold_i,(t,v) in enumerate(kfold.split(y_train))
    )
    for split_i, eta_i, mae in mae_generator:
        maes[split_i, eta_i] = mae

    concat_results = np.full((len(eta_list), 3), np.inf)
    for eta_i in range(len(eta_list)):
        res = maes[:,eta_i]
        #res = res[np.isfinite(res)]
        concat_results[eta_i,0] = res.mean()
        concat_results[eta_i,1] = res.std()
        concat_results[eta_i,2] = eta_list[eta_i]
    return concat_results

def search_sigma(
    X_train, y_train, splits,
    kernel, sigma, eta_grid,
    sparse_idx=None,
    n_sigma_iter=5, stddev_portion=0.0,
    adaptive=False, adaptive_v2=False,
    read_kernel=False, printlevel=0,
):
    """Search the optimal values of sigma and alpha for a KRR model with known representations.

    Sigma is the width parameter of the kernel function used,
    and alpha is the regularisation parameter of the resulting matrix equation.

    Internally, this can call for either a simple grid search, or be modified as so:
    - the grid is adaptative for sigma (adaptive)
    - the grid search for sigma becomes a continuous line search (adaptive_v2)
    No matter what, the optimisation of alpha is done over a grid, with k-fold validation.

    Args:
        X_train (np.ndarray[n_total,n_features]): feature vectors for the combined train-validation dataset
        y_train (np.ndarray[n_total]): learnable properties for all inputs
        sigma (array-like(float)): values of sigma. for `adaptive`, starting values, for `adaptive_v2`, only the first and last values are used, as presumed bounds of the optimal value of sigma
        eta_grid (array-like of floats): values of eta to try
        n_sigma_iter (int): number of iterations for the sigma line-search (if adaptive_v2)
        kernel (callable): kernel function computing a kernel matrix from two sets of representations vectors and a "gamma" scale parameter
        splits (int): number of folds for k-fold validation
        stddev_portion (float): contribution of the error's standard deviation to compare error distributions
        sparse_idx (optional np.ndarray[int, n_references]): selection of reference inputs for sparse KRR.
        adaptive (bool): to use the adaptive grid for sigma
        adaptive_v2 (bool): to use the line search for sigma
        read_kernel (bool): to completely discard sigma, assuming the representation array is a precomputed kernel array
        printlevel (int): level of verbosity

    Returns:
        sigma (float): optimal value of sigma
        alpha (float): optimal value of alpha
        costs (np.ndarray[n_splits]): validation error distribution for these values of sigma,alpha
    """
    errors = []

    def get_err(s):
        if read_kernel is False:
            K_all = kernel(X_train, X_train, 1.0/s)
        else:
            K_all = X_train

        sparse = sparse_idx is not None
        if sparse:
            K_all = K_all[:,sparse_idx]

        results_per_eta = kfold_alpha_eval(
            K_all, y_train, splits, eta_grid, sparse,
            parallel = parallel,
        )
        for mean,std,e in results_per_eta:
            if printlevel>0 :
                sys.stderr.flush()
                print(s, e, mean, std, flush=True)
            errors.append((mean, std, e, s))

        costs = results_per_eta[:,0] +  stddev_portion*results_per_eta[:,1]
        return costs.min()

    with Parallel(n_jobs=-1) as parallel:
        if adaptive_v2:
            if adaptive:
                raise ValueError("Only one of `adaptive`, `adaptive_v2` may be specified.")
            _, _ = parabolic_search(
                np.log(sigma[0]), np.log(sigma[-1]),
                lambda log_s: get_err(np.exp(log_s)),
                n_iter=n_sigma_iter, x_thres=0.1, y_thres=0.01,
            )
        elif adaptive:
            _ = adaptative_grid_search(sigma, get_err)
        else:
            for s in sigma:
                get_err(s)

    return np.asarray(errors)

def hyperparameters(X, y,
           sigma=defaults.sigmaarr, eta=defaults.etaarr, sparse=None,
           akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict, read_kernel=False,
           test_size=defaults.test_size, splits=defaults.splits, idx_test=None, idx_train=None,
           stddev_portion=0.0, n_sigma_iters=5,
           printlevel=0, adaptive=False, adaptive_v2=False, random_state=defaults.random_state,
):
    """Perform a Kfold cross-validated hyperparameter optimization (for width of kernel and regularization parameter).

    Args:
        X (numpy.ndarray[Nsamples,...]): Array containing the representations of all Nsamples.
        y (numpy.1darray[Nsamples]): Array containing the target property of all Nsamples.
        sigma (list): List of kernel width for the grid search.
        eta (list): List of regularization strength for the grid search.
        sparse (int): The number of reference environnments to consider for sparse regression.
        akernel (str): Local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine').
        gkernel (str): Global kernel (None, 'REM', 'avg').
        gdict (dict): Parameters of the global kernels.
        read_kernel (bool): If 'X' is a kernel and not an array of representations.
        test_size (float or int): Test set fraction (or number of samples).
        splits (int): K number of splits for the Kfold cross-validation.
        idx_test (numpy.1darray): List of indices for the test set (based on the sequence in X).
        idx_train (numpy.1darray): List of indices for the training set (based on the sequence in X).
        stddev_portion (float): The amount of error standard deviation to add to error means, for error distribution ranking.
        n_sigma_iters (int): for adaptive_v2, the number of iterations to run the sigma line search for.
        printlevel (int): Controls level of output printing.
        adaptive (bool): To expand the grid search adaptatively.
        adaptive_v2 (bool): To optimise sigma though line search rather than grid search, using the ends of the `sigma` list as presumed lower/upper bounds for the optimal value.
        random_state (int): The seed used for random number generator (controls train/test splitting).

    Returns:
        The results of the grid search as a numpy.2darray [Cx(MAE,std,eta,sigma)],
            where C is the number of parameter set and
            the array is sorted according to MAEs (last is minimum)

    Raises:
        RuntimeError: If 'X' is a kernel and sparse regression is chosen.
    """
    if gkernel is None:
        gwrap = None
    else:
        gwrap = [gkernel, gdict]
    kernel = get_kernel(akernel, gwrap)

    idx_train, _, y_train, _ = train_test_split_idx(y=y, idx_test=idx_test, idx_train=idx_train,
                                                    test_size=test_size, random_state=random_state)
    if read_kernel is False:
        X_train = X[idx_train]
        optimise_sigma = True
    else:
        X_train = X[np.ix_(idx_train,idx_train)]
        optimise_sigma = False

    if sparse:
        if read_kernel:
            raise RuntimeError('Cannot do FPS with kernels')
        sparse_idx = do_fps(X_train)[0][:sparse]
    else:
        sparse_idx = None

    with Parallel(n_jobs=1, return_as="generator_unordered") as parallel:
        if optimise_sigma:
            errors = search_sigma(
                X_train, y_train, splits,
                kernel, sigma, eta, sparse_idx,
                n_sigma_iters, stddev_portion,
                adaptive, adaptive_v2,
                read_kernel, printlevel,
            )
        else:
            K_all = X_train
            sparse = sparse_idx is not None
            if sparse:
                K_all = K_all[:, sparse_idx]
            sigma = np.nan
            partial_errors = kfold_alpha_eval(
                K_all, y, splits, eta,
                sparse, parallel,
            )
            errors = np.ndarray((len(partial_errors),4))
            errors[:,:3] = partial_errors
            errors[:,3] = np.nan

    sorter = (errors[:,0] + stddev_portion*errors[:,1]).argsort()
    errors = errors[sorter[::-1]]

    # work_sigma = sigma
    # errors = []
    # direction = None
    # while True:
    #     errors = list(errors)
    #     errors.extend(hyper_loop(work_sigma, eta))
    #     errors = np.array(errors)
    #     ind = np.argsort(errors[:,0])[::-1]
    #     errors = errors[ind]

    #     if not adaptive:
    #         break

    #     best_sigma = errors[-1][3]
    #     new_sigma = None

    #     if direction is None:
    #         if   best_sigma==max(work_sigma):
    #             direction = 'up'
    #         elif best_sigma==min(work_sigma):
    #             direction = 'down'

    #     # at the 1st iteration if is checked twice on purpose
    #     if direction=='up'     and best_sigma==max(work_sigma):
    #         new_sigma = best_sigma*np.array(defaults.sigmaarr_mult[1:])
    #     elif direction=='down' and best_sigma==min(work_sigma):
    #         new_sigma = best_sigma/np.array(defaults.sigmaarr_mult[1:])

    #     if new_sigma is None:
    #         break
    #     work_sigma = new_sigma
    #     print('next iteration:', work_sigma, flush=True)
    return errors


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = RegressionParser(description='This program finds the optimal hyperparameters.', hyperparameters_set='array')
    parser.remove_argument("train_size")
    parser.add_argument('--ada2',    action='store_true', dest='adaptive_v2',    default=False, help='whether to use a continuous adaptative approach to sigma. If so, only the first and last sigma values will be used to start the optimisation.')
    parser.add_argument('--stddev-portion',   type=float, dest='stddev_portion', default=0.0,   help='amount of error standard deviation to add to error means, for error distribution ranking in the output.')
    parser.add_argument('--sigma-iterations', type=int,   dest='n_sigma_iters',  default=5,     help='number of iterations for the sigma-optimisation line search')


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

    errors = hyperparameters(
        X, y,
        sigma=args.sigma, eta=args.eta, sparse=args.sparse,
        akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict, read_kernel=args.readk,
        test_size=args.test_size, splits=args.splits, idx_test=None, idx_train=None,
        stddev_portion=args.stddev_portion, n_sigma_iters=args.n_sigma_iters,
        printlevel=args.printlevel, adaptive=args.adaptive, adaptive_v2=args.adaptive_v2,
        random_state=args.random_state,
    )
    errors = np.array(errors)
    if args.nameout is not None:
        np.savetxt(args.nameout, errors, header="error        stdev          eta          sigma")
    print()
    print('error        stdev          eta          sigma')
    for error in errors:
        print("{:e} {:e} | {:e} {:e}".format(*tuple(error)))


if __name__ == "__main__":
    main()
