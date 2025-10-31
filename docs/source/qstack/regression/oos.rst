qstack.regression.oos
=====================

Functions
---------

oos (X, X\_oos, alpha, sigma=defaults.sigma, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict, test\_size=defaults.test\_size, idx\_test=None, idx\_train=None, sparse=None, random\_state=defaults.random\_state)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Perform prediction on an out-of-sample (OOS) set.

    Args:
        X (numpy.2darray[Nsamples,Nfeat]): array containing the 1D representations of all Nsamples
        X_oos (numpy.2darray[Noos,Nfeat]): array of OOS representations.
        alpha (numpy.1darray(Ntrain or sparse)): regression weights.
        sigma (float): width of the kernel
        akernel (str): local kernel (Laplacian, Gaussian, linear)
        gkernel (str): global kernel (REM, average)
        gdit (dict): parameters of the global kernels
        test_size (float or int): test set fraction (or number of samples)
        random_state (int): the seed used for random number generator (controls train/test splitting)
        idx_test (list): list of indices for the test set (based on the sequence in X)
        idx_train (list): list of indices for the training set (based on the sequence in X)
        sparse (int): the number of reference environnments to consider for sparse regression

    Returns:
        np.1darray(Noos) : predictions on the OOS set

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
