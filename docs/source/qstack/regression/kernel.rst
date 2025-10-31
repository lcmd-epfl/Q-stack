qstack.regression.kernel
========================

Functions
---------

kernel (X, Y=None, sigma=defaults.sigma, akernel=defaults.kernel, gkernel=defaults.gkernel, gdict=defaults.gdict)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes a kernel between sets A and B (or A and A) using their representations.

    Args:
        X (list of arrays): Representation of A
        Y (list of arrays): Representation of B.
        sigma (): Sigma hyperparameter.
        akernel (): Kernel type (G for Gaussian, L for Laplacian, and myL for Laplacian for open-shell systems).
        gkernel (): Global kernel type (agv for average, rem for REMatch kernel, None for local kernels).
        gdict (): Dictionary like input string to initialize global kernel parameters. Defaults to {'alpha':1.0, 'normalize':1}.

    Returns:
        A numpy ndarray containing the kernel.

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
