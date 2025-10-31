qstack.basis\_opt.opt
=====================

Functions
---------

optimize\_basis (elements\_in, basis\_in, molecules\_in, gtol\_in=1e-07, method\_in='CG', printlvl=2, check=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Optimize a given basis set.

    Args:
        elements_in (str):
        basis_in (str or dict): Basis set
        molecules_in (dict): which contains the cartesian coordinates of the molecule (string) with the key 'atom', the uncorrelated on-top pair density on a grid (numpy array) with the key 'rho', the grid coordinates (numpy array) with the key 'coords', and the grid weights (numpy array) with the key 'weight'.
        gtol_in (float): Gradient norm must be less than gtol_in before successful termination (minimization).
        method_in (str): Type of solver. Check scipy.optimize.minimize for full documentation.
        printlvl (int):
        check (bool):

    Returns:
        Dictionary containing the optimized basis.

main ()
~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
