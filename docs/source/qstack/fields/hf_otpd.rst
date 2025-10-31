qstack.fields.hf\_otpd
======================

Functions
---------

hf\_otpd (mol, dm, grid\_level=3, save\_otpd=False, return\_all=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the uncorrelated on-top pair density on a grid.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Density matrix in AO-basis.
        grid_level (int): Controls the number of radial and angular points.
        save_otpd (bool): If True, saves the input and output in a .npz file. Defaults to False
        return_all (bool): If true, returns the uncorrelated on-top pair density on a grid, and the cooresponding pyscf Grid object; if False, returns only the uncorrelated on-top pair density. Defaults to False

    Returns:
        A numpy ndarray with the uncorrelated on-top pair density on a grid. If 'return_all' = True, then it also returns the corresponding pyscf Grid object.

save\_OTPD (mol, otpd, grid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Saves the information about an OTPD computation into a .npz file.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        otpd (numpy ndarray): On-top pair density on a grid.
        grid (pyscf Grid): Grid object

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
