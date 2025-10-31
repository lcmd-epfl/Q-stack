qstack.regression.cross\_validate\_results
==========================================

Functions
---------

cv\_results (X, y, sigmaarr=defaults.sigmaarr, etaarr=defaults.etaarr, gkernel=defaults.gkernel, gdict=defaults.gdict, akernel=defaults.kernel, test\_size=defaults.test\_size, train\_size=defaults.train\_size, splits=defaults.splits, printlevel=0, adaptive=False, read\_kernel=False, n\_rep=defaults.n\_rep, save=False, preffix='unknown', save\_pred=False, progress=False, sparse=None, seed0=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes various learning curves (LC) ,with random sampling, and returns the average performance.

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

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
