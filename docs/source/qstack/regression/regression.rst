qstack.regression.regression
============================

Functions
---------

regression (X, y, read\_kernel=False, sigma=defaults.sigma, eta=defaults.eta, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict, test\_size=defaults.test\_size, train\_size=defaults.train\_size, n\_rep=defaults.n\_rep, random\_state=defaults.random\_state, idx\_test=None, idx\_train=None, sparse=None, debug=False, save\_pred=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Produces learning curves (LC) data, for various training sizes, using kernel ridge regression and the user specified parameters

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

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
