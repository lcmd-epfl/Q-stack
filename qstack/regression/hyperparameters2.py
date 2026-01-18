"""Hyperparameter optimisation using a smoother version of sigma selection,
using a no-gradient line search"""

import sys, logging
import numpy as np
import scipy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.utils.parallel import Parallel, delayed
from qstack.mathutils.fps import do_fps
from qstack.tools import correct_num_threads
from .kernel_utils import get_kernel, defaults, train_test_split_idx
from .parser import RegressionParser

logger = logging.getLogger("qstack.regression.hyperparameters2")

# #####################
# parabola-based line search


def fit_quadratic(x1,x2,x3, y1,y2,y3):
    """Compute the three coefficients of a quadratic polynomial going through three given points."""
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
    """A 1D optimisation function, assuming the loss function in question is convex, It first checks to see the bounds are correct, then refines them by fitting quadratic polynomials"""

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
    logger.debug('chosen: %f\%f/%f', x_left, x_center, x_right)
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
        logger.debug('predicted local minimum at %f->%f, true error %f', x_new, ypred_new, y_new)
        all_errs.append((x_new, y_new)) ; all_errs.sort()
        logger.debug('current data: %r', all_errs)

        if x_new < x_left or x_new > x_right:
            logger.debug('predicted local minimum not in immediate bounds, regaining bearings...')
            new_index = np.argmin(np.asarray(all_errs)[:,1])
            if new_index in (0, len(all_errs)-1):
                raise AssertionError('edges of the search are somehow the minimum in second phase of function')
            x_left, y_left = all_errs[new_index-1]
            x_right, y_right = all_errs[new_index+1]
            x_center, y_center = all_errs[new_index]

        elif y_new > y_center:
            if x_new > x_center:
                x_right, y_right = x_new, y_new
            else:
                x_left, y_left = x_new, y_new
        elif y_left < y_right:
            if max(y_right,y_left, y_new)-min(y_new, y_center) < y_thres:
                break
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



def kfold_alpha_eval(K_all, y, n_splits, alpha_grid, parallel=None, on_compute=(lambda eta,err,stderr:None)):
    if parallel is None:
        parallel = Parallel(n_jobs=-1, return_as="generator_unordered")
    kfold = KFold(n_splits=n_splits, shuffle=False)
    maes = np.full((kfold.get_n_splits(), len(alpha_grid)), np.inf)
    y = np.asarray(y)
    is_sparse = K_all.shape[0] != K_all.shape[1]

    def inner_loop(split_i, alpha_i, train_idx, val_idx, alpha):
        y_val = y[val_idx]
        y_train = y[train_idx]
        if is_sparse:
            K_train = K_all[train_idx, :]
            K_val = K_all[val_idx, :]
            y_train = K_train.T @ y_train
            K_train = K_train.T @ K_train
        else:
            K_train = K_all[np.ix_(train_idx,train_idx)]
            K_val = K_all [np.ix_(val_idx,train_idx)]
            if np.may_share_memory(K_train, K_all):
                K_train = K_train.copy()
        K_train[np.diag_indices_from(K_train)] += alpha

        try:
            weights = scipy.linalg.solve(K_train, y_train, assume_a='pos', overwrite_a=True)
        except Exception as err:
            raise
            # bad fit (singular matrix)!
            return split_i, alpha_i, np.inf
        predict = K_val @ weights
        return split_i, alpha_i, mean_absolute_error(y_val, predict)

    mae_generator = parallel(
        delayed(inner_loop)(s_i,a_i, t,v,a)
        for a_i,a in enumerate(alpha_grid)
        for s_i,(t,v) in enumerate(kfold.split(y))
    )
    for split_i, alpha_i, mae in mae_generator:
        maes[split_i, alpha_i] = mae

    concat_results = np.full((len(alpha_grid), 2), np.inf)
    for alpha_i in range(len(alpha_grid)):
        if not np.isfinite(maes[:, alpha_i]).any():
            pass
        else:
            res = maes[alpha_i]
            res = res[np.isfinite(res)]
            concat_results[alpha_i,0] = res.mean()
            concat_results[alpha_i,1] = res.std()
            on_compute(alpha_grid[alpha_i], *concat_results[alpha_i])
    #print("kfold evaluation for alpha grid",alpha_grid,concat_results)
    selected_alpha_i = concat_results[:,0].argmin()
    return alpha_grid[selected_alpha_i], maes[:,selected_alpha_i]#, models[:, selected_alpha_i].copy()


def search_sigma(
    X, y, kernel,
    sigma_bounds, alpha_grid,
    n_iter, n_splits,
    stddev_portion=+1.0, sparse_idx=None,
    parallel=None, on_compute=(lambda sigma,eta,err,stderr:None)
):
    """Search"""

    sigma_left, sigma_right = sigma_bounds

    err_dict = {}

    def get_err(log_sigma):
        sigma = np.exp(log_sigma)

        K_all = kernel(X, X, 1.0/sigma)
        if sparse_idx is not None:
            K_all = K_all[:, sparse_idx]

        alpha, costs = kfold_alpha_eval(
            K_all, y, n_splits, alpha_grid,
            parallel=parallel, on_compute=(lambda eta,err,stderr: on_compute(sigma,eta,err,stderr)),
        )
        err_dict[log_sigma] = (alpha,costs)
        cost_res = costs.mean() + stddev_portion*costs.std()
        return cost_res

    log_sigma_selected, cost_selected = parabolic_search(
        np.log(sigma_left), np.log(sigma_right),
        get_err,
        n_iter=n_iter, x_thres=0.1, y_thres=0.01,
    )

    alpha_selected, costs_selected = err_dict[log_sigma_selected]
    sigma = np.exp(log_sigma_selected)

    return sigma, alpha_selected, costs_selected



def hyperparameters(X, y,
           sigma_low=defaults.sigmaarr[0], sigma_high=defaults.sigmaarr[-1], eta=defaults.etaarr,
           akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict, read_kernel=False,
           test_size=defaults.test_size, splits=defaults.splits, n_sigma_iters=5, idx_test=None, idx_train=None,
           printlevel=0, sparse=None,
           stddev_portion=+1.0,
           random_state=defaults.random_state,
):
    """Perform a Kfold cross-validated hyperparameter optimization (for width of kernel and regularization parameter).

    Args:
        X (numpy.ndarray[Nsamples,...]): Array containing the representations of all Nsamples.
        y (numpy.1darray[Nsamples]): Array containing the target property of all Nsamples.
        sigma_low (float): Estimated low bound forthe kernel width.
        sigma_high (float): Estimated high bound forthe kernel width.
        eta (list): List of regularization strength for the grid search.
        akernel (str): Local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine').
        gkernel (str): Global kernel (None, 'REM', 'avg').
        gdict (dict): Parameters of the global kernels.
        test_size (float or int): Test set fraction (or number of samples).
        splits (int): K number of splits for the Kfold cross-validation.
        n_sigma_iters (int): number of iterations for the sigma-optimisation line search
        idx_test (numpy.1darray): List of indices for the test set (based on the sequence in X).
        idx_train (numpy.1darray): List of indices for the training set (based on the sequence in X).
        printlevel (int): Controls level of output printing.
        read_kernel (bool): If 'X' is a kernel and not an array of representations (disables sigma optimisation).
        sparse (int): The number of reference environnments to consider for sparse regression.
        stddev_portion (float): The amount of error standard deviation to add to error means, for error distribution ranking.
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
        sparse_idx = np.arange(X_train.shape[0])

    errors = []
    with Parallel(n_jobs=-1, return_as="generator_unordered") as parallel:
        if optimise_sigma:
            err_append = lambda sigma,alpha,err,stderr: errors.append((err,stderr, alpha,sigma))
            _,_,_ = search_sigma(
                X_train, y_train, kernel, (sigma_low, sigma_high), alpha_grid=eta,
                parallel=parallel, on_compute=err_append,
                n_iter = n_sigma_iters, n_splits=splits, stddev_portion=stddev_portion,
                sparse_idx=sparse_idx,
            )
        else:
            if sparse_idx is not None:
                K_all = X_train[:, sparse_idx]
            else:
                K_all = X_train
            sigma = np.nan
            err_append = lambda alpha,err,stderr: errors.append((err,stderr, alpha,sigma))
            _,_ = kfold_alpha_eval(
                K_all, y, splits, alpha_grid=eta,
                parallel=parallel, on_compute=err_append,
            )


    errors = np.array(errors)
    ind = np.argsort(errors[:,0]+stddev_portion*errors[:,-1])[::-1]
    errors = errors[ind]
    return errors


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = RegressionParser(description='This program finds the optimal hyperparameters.', hyperparameters_set='array')
    parser.remove_argument("train_size")
    parser.remove_argument("sigma")
    parser.remove_argument("adaptative")
    parser.add_argument('--sigma-low',        type=float, dest='sigma_low',      default=1E-2, help='estimated low bound for sigma')
    parser.add_argument('--sigma-high',       type=float, dest='sigma_high',     default=1E+2, help='estimated high bound for sigma')
    parser.add_argument('--stddev-portion',   type=float, dest='stddev_portion', default=1,    help='amount of error standard deviation to add to error means, for error distribution ranking in the output.')
    parser.add_argument('--sigma-iterations', type=int,   dest='n_sigma_iters',  default=5,    help='number of iterations for the sigma-optimisation line search')


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
        X, y, read_kernel=args.readk,
        sigma_low=args.sigma_low, sigma_high=args.sigma_high, eta=args.eta,
        akernel=args.akernel, gkernel=args.gkernel, gdict=args.gdict, sparse=args.sparse,
        test_size=args.test_size, splits=args.splits, n_sigma_iters=args.n_sigma_iters,
        printlevel=args.printlevel, random_state=args.random_state, stddev_portion=args.stddev_portion,
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
