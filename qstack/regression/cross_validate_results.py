import numpy as np
from tqdm import tqdm
from qstack.tools import correct_num_threads
from .kernel_utils import defaults
from .hyperparameters import hyperparameters
from .regression import regression
from .parser import RegressionParser


def cv_results(X, y,
            sigmaarr=defaults.sigmaarr, etaarr=defaults.etaarr, akernel=defaults.kernel,
            gkernel=defaults.gkernel, gdict=defaults.gdict, test_size=defaults.test_size,
            train_size=defaults.train_size, splits=defaults.splits, printlevel=0,
            adaptive=False, read_kernel=False, n_rep=defaults.n_rep, save=False,
            preffix='unknown', save_pred=False, progress=False, sparse=None,
            seed0=0):
    """ Computes various learning curves (LC) ,with random sampling, and returns the average performance.

    Args:
        X (numpy.ndarray[Nsamples,...]): array containing the representations of all Nsamples
        y (numpy.1darray[Nsamples]): array containing the target property of all Nsamples
        sigmaarr (list): list of kernel width for the grid search
        etaarr (list): list of regularization strength for the grid search
        akernel (str): local kernel ('L' for Laplacian, 'G' for Gaussian, 'dot', 'cosine')
        gkernel (str): global kernel (None, 'REM', 'avg')
        gdict (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        train_size (list): list of training set size fractions used to evaluate the points on the LC
        splits (int): K number of splits for the Kfold cross-validation
        printlevel (int): controls level of output printing
        adaptative (bool): to expand the grid for optimization adaptatively
        read_kernel (bool): if 'X' is a kernel and not an array of representations
        n_rep (int): the number of repetition for each point (using random sampling)
        save (bool): wheather to save intermediate LCs (.npy)
        preffix (str): the prefix to use for filename when saving intemediate results
        save_pred (bool): to save predicted targets for all LCs (.npy)
        progress (bool): to print a progress bar
        sparse (int): the number of reference environnments to consider for sparse regression
        seed0 (int): the initial seed to produce a set of seeds used for random number generator

    Returns:
        The averaged LC data points as a numpy.ndarray containing (train sizes, MAE, std)
    """
    hyper_runs = []
    lc_runs = []
    seeds = seed0+np.arange(n_rep)
    if save_pred:
        predictions_n = []
    for seed in tqdm(seeds, disable=(not progress)):
        error = hyperparameters(X, y, read_kernel=read_kernel, sigma=sigmaarr, eta=etaarr,
                                gkernel=gkernel, gdict=gdict,
                                akernel=akernel, test_size=test_size, splits=splits,
                                printlevel=printlevel, adaptive=adaptive, random_state=seed,
                                sparse=sparse)
        _mae, _stdev, eta, sigma = zip(*error, strict=True)
        maes_all = regression(X, y, read_kernel=read_kernel, sigma=sigma[-1], eta=eta[-1],
                              gkernel=gkernel, gdict=gdict,
                              akernel=akernel, test_size=test_size, train_size=train_size,
                              n_rep=1, debug=True, save_pred=save_pred,
                              sparse=sparse, random_state=seed)
        if save_pred:
            res, pred = maes_all[1]
            maes_all = maes_all[0]
            predictions_n.append((res,pred))
        ind = np.argsort(error[:,3])
        error = error[ind]
        ind = np.argsort(error[:,2])
        error = error[ind]
        hyper_runs.append(error)
        lc_runs.append(maes_all)
    lc_runs = np.array(lc_runs)
    hyper_runs = np.array(hyper_runs, dtype=object)
    lc = list(zip(lc_runs[:,:,0].mean(axis=0), lc_runs[:,:,1].mean(axis=0), lc_runs[:,:,1].std(axis=0), strict=True))
    lc = np.array(lc)
    if save:
        np.save(f"{preffix}_{n_rep}-hyper-runs.npy", hyper_runs)
        np.save(f"{preffix}_{n_rep}-lc-runs.npy", lc_runs)
    if save_pred:
        np_pred = np.array(predictions_n)
        ##### Can not take means !!! Test-set varies with run !
        ##### pred_mean = np.concatenate([np_pred.mean(axis=0),np_pred.std(axis=0)[1].reshape((1,-1))], axis=0)
        pred_mean = np.concatenate([*np_pred.reshape((n_rep, 2, -1))], axis=0)
        np.savetxt(f"{preffix}_{n_rep}-predictions.txt", pred_mean.T)
    return lc


def main():
    """Command-line entry point for full cross-validation with hyperparameter search."""
    parser = RegressionParser(description='This program runs a full cross-validation of the learning curves (hyperparameters search included).', hyperparameters_set='array')
    parser.remove_argument('random_state')
    parser.add_argument('--n',          type=int,            dest='n_rep',     default=defaults.n_rep,  help='the number of repetition for each point')
    parser.add_argument('--save',       action='store_true', dest='save_all',  default=False,           help='if saving intermediate results in .npy file')
    parser.add_argument('--save-pred',  action='store_true', dest='save_pred', default=False,           help='if save test-set prediction')

    args = parser.parse_args()
    if(args.readk):
        args.sigma = [np.nan]
    if(args.ll):
        correct_num_threads()

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    print(vars(args))
    final = cv_results(X, y, sigmaarr=args.sigma, etaarr=args.eta,
                       gdict=args.gdict, gkernel=args.gkernel, akernel=args.akernel,
                       read_kernel=args.read_kernel,
                       test_size=args.test_size, splits=args.splits, printlevel=args.printlevel,
                       adaptive=args.adaptive, train_size=args.train_size, n_rep=args.n_rep,
                       preffix=args.nameout, save=args.save_all, save_pred=args.save_pred,
                       sparse=args.sparse, progress=True)
    print(final)
    np.savetxt(args.nameout+'.txt', final)


if __name__ == '__main__':
    main()
