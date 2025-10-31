qstack.regression.final\_error
==============================

Functions
---------

final\_error (X, y, read\_kernel=False, sigma=defaults.sigma, eta=defaults.eta, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict, test\_size=defaults.test\_size, idx\_test=None, idx\_train=None, sparse=None, random\_state=defaults.random\_state, return\_pred=False, return\_alpha=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Perform prediction on the test set using the full training set.

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
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (list): list of indices for the test set (based on the sequence in X)
        idx_train (list): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression
        return_pred (bool) : return predictions
        return_alpha (bool) : return regression weights

    Returns:
        np.1darray(Ntest) : prediction absolute errors on the test set
        np.1darray(Ntest) : (if return_pred is True) predictions on the test set
        np.1darray(Ntrain or sparse) : (if return_alpha is True) regression weights

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
