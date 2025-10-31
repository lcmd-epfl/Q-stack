qstack.regression.hyperparameters
=================================

Functions
---------

hyperparameters (X, y, sigma=defaults.sigmaarr, eta=defaults.etaarr, gkernel=defaults.gkernel, gdict=defaults.gdict, akernel=defaults.kernel, test\_size=defaults.test\_size, splits=defaults.splits, idx\_test=None, idx\_train=None, printlevel=0, adaptive=False, read\_kernel=False, sparse=None, random\_state=defaults.random\_state)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Performs a Kfold cross-validated hyperparameter optimization (for width of kernel and regularization parameter).

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        y (numpy.1darray[Nsamples]): array containing the target property of all Nsamples
        sigma (list): list of kernel width for the grid search
        eta (list): list of regularization strength for the grid search
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        akernel (str): local kernel (Laplacian, Gaussian, linear)
        test_size (float or int): test set fraction (or number of samples)
        splits (int): K number of splits for the Kfold cross-validation
        idx_test (list): list of indices for the test-set (based on the sequence in X
        idx_train (list): list of indices for the training set (based on the sequence in X)
        printlevel (int): controls level of output printing
        adaptative (bool): to expand the grid search adaptatively
        read_kernel (bool): if 'X' is a kernel and not an array of representations
        sparse (int): the number of reference environnments to consider for sparse regression
        random_state (int): the seed used for random number generator (controls train/test splitting)

    Returns:
        The results of the grid search as a numpy.2darray [Cx(MAE,std,eta,sigma)],
        where C is the number of parameter set and
        the array is sorted according to MAEs (last is minimum)

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
