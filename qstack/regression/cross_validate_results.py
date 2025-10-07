#!/usr/bin/env python3

import numpy as np
from .kernel_utils import defaults, ParseKwargs
from .hyperparameters import hyperparameters
from .regression import regression


def cv_results(X, y,
            sigmaarr=defaults.sigmaarr, etaarr=defaults.etaarr, gkernel=defaults.gkernel,
            gdict=defaults.gdict, akernel=defaults.kernel, test_size=defaults.test_size,
            train_size=defaults.train_size, splits=defaults.splits, printlevel=0,
            adaptive=False, read_kernel=False, n_rep=defaults.n_rep, save=False,
            preffix='unknown', save_pred=False, progress=False, sparse=None,
            seed0=0):
    """ Computes various learning curves (LC) ,with random sampling, and returns the average performance.

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        y (numpy.1darray[Nsamples]): array containing the target property of all Nsamples
        sigmaar (list): list of kernel widths for the hyperparameter optimization
        etaar (list): list of regularization strength for the hyperparameter optimization
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        akernel (str): local kernel (Laplacian, Gaussian, linear)
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
    if save_pred: predictions_n = []
    if progress:
        import tqdm
        seeds = tqdm.tqdm(seeds)
    for seed,n in zip(seeds, range(n_rep)):
        error = hyperparameters(X, y, read_kernel=False, sigma=sigmaarr, eta=etaarr,
                                akernel=akernel, test_size=test_size, splits=splits,
                                printlevel=printlevel, adaptive=adaptive, random_state=seed,
                                sparse=sparse)
        mae, stdev, eta, sigma = zip(*error)
        maes_all = regression(X, y, read_kernel=False, sigma=sigma[-1], eta=eta[-1],
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
    lc = list(zip(lc_runs[:,:,0].mean(axis=0), lc_runs[:,:,1].mean(axis=0), lc_runs[:,:,1].std(axis=0)))
    lc = np.array(lc)
    if save == True:
        np.save(f"{preffix}_{n_rep}-hyper-runs.npy", hyper_runs)
        np.save(f"{preffix}_{n_rep}-lc-runs.npy", lc_runs)
    if save_pred == True:
        np_pred = np.array(predictions_n)
        ##### Can not take means !!! Test-set varies with run !
        ##### pred_mean = np.concatenate([np_pred.mean(axis=0),np_pred.std(axis=0)[1].reshape((1,-1))], axis=0)
        pred_mean = np.concatenate([*np_pred.reshape((n_rep, 2, -1))], axis=0)
        np.savetxt(f"{preffix}_{n_rep}-predictions.txt", pred_mean.T)
    return lc


def main():
    import argparse
    from qstack.tools import correct_num_threads
    parser = argparse.ArgumentParser(description='This program runs a full cross-validation of the learning curves (hyperparameters search inbcluded).')
    parser.add_argument('--x',      type=str,   dest='repr',       required=True, help='path to the representations file')
    parser.add_argument('--y',      type=str,   dest='prop',       required=True, help='path to the properties file')
    parser.add_argument('--test',   type=float, dest='test_size',  default=defaults.test_size, help='test set fraction (default='+str(defaults.test_size)+')')
    parser.add_argument('--train',      type=float, dest='train_size', default=defaults.train_size, nargs='+', help='training set fractions')
    parser.add_argument('--akernel',     type=str,   dest='akernel',     default=defaults.kernel,    help='local kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel',     type=str,   dest='gkernel',     default=defaults.gkernel,    help='global kernel type (avg for average kernel, rem for REMatch kernel) (default )')
    parser.add_argument('--gdict',     nargs='*',   action=ParseKwargs, dest='gdict',     default=defaults.gdict,    help='dictionary like input string to initialize global kernel parameters')
    parser.add_argument('--splits', type=int,   dest='splits',     default=defaults.splits,    help='k in k-fold cross validation (default='+str(defaults.n_rep)+')')
    parser.add_argument('--n', type=int,   dest='n_rep',     default=defaults.n_rep,    help='k in k-fold cross validation (default='+str(defaults.n_rep)+')')
    parser.add_argument('--print',  type=int,   dest='printlevel', default=0,                  help='printlevel')
    parser.add_argument('--eta',    type=float, dest='eta',   default=defaults.etaarr,   nargs='+', help='eta array')
    parser.add_argument('--sigma',  type=float, dest='sigma', default=defaults.sigmaarr, nargs='+', help='sigma array')
    parser.add_argument('--ll',   action='store_true', dest='ll',       default=False,  help='if correct for the numper of threads')
    parser.add_argument('--save',   action='store_true', dest='save_all',       default=False,  help='if saving intermediate results in .npy file')
    parser.add_argument('--ada',  action='store_true', dest='adaptive', default=False,  help='if adapt sigma')
    parser.add_argument('--save-pred',  action='store_true', dest='save_pred', default=False,  help='if save test-set prediction')
    parser.add_argument('--readkernel', action='store_true', dest='readk', default=False,  help='if X is kernel')
    parser.add_argument('--sparse',     type=int, dest='sparse', default=None,  help='regression basis size for sparse learning')
    parser.add_argument('--name',      type=str,   dest='nameout',       required=True, help='the name of the output file')
    args = parser.parse_args()
    if(args.readk): args.sigma = [np.nan]
    if(args.ll): correct_num_threads()

    X = np.load(args.repr)
    y = np.loadtxt(args.prop)
    print(vars(args))
    final = cv_results(X, y, sigmaarr=args.sigma, etaarr=args.eta, akernel=args.akernel,
                       test_size=args.test_size, splits=args.splits, printlevel=args.printlevel,
                       adaptive=args.adaptive, train_size=args.train_size, n_rep=args.n_rep,
                       preffix=args.nameout, save=args.save_all, save_pred=args.save_pred,
                       sparse=args.sparse, progress=True)
    print(final)
    np.savetxt(args.nameout+'.txt', final)


if __name__ == '__main__':
    main()
