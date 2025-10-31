qstack.fields.hirshfeld
=======================

Functions
---------

spherical\_atoms (elements, atm\_bas)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Get density matrices for spherical atoms.

    Args:
        elements (list of str): Elements to compute the DM for.
        atm_bas (string / pyscf basis dictionary): Basis to use.

    Returns:
        A dict of numpy 2d ndarrays which contains the atomic density matrices for each element with its name as a key.

\_hirshfeld\_weights (mol\_full, grid\_coord, atm\_dm, atm\_bas, dominant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the Hirshfeld weights.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        grid_coord (numpy ndarray): Coordinates of the grid.
        dm_atoms (dict of numpy 2d ndarrays): Atomic density matrices (output of the `spherical_atoms` fn).
        atm_bas (string / pyscf basis dictionary): Basis set used to compute dm_atoms.
        dominant (bool): Whether to use dominant or classical partitioning.

    Returns:
        A numpy ndarray containing the computed Hirshfeld weights.

hirshfeld\_charges (mol, cd, dm\_atoms=None, atm\_bas=None, dominant=True, occupations=False, grid\_level=3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        cd (1D or 2D numpy ndarray or list of arrays): Density-fitting coefficients / density matrices.
        dm_atoms (dict of numpy 2d ndarrays): Atomic density matrices (output of the `spherical_atoms` fn).
                                              If None, is computed on-the-fly.
        atm_bas (string / pyscf basis dictionary): Basis set used to compute dm_atoms.
                                                   If None, is taken from mol.
        dominant (bool): Whether to use dominant or classical partitioning.
        occupations (bool): Whether to return atomic occupations or charges.
        grid level (int): Grid level for numerical integration.

    Returns:
        A numpy 1d ndarray or list of them containing the computed atomic charges or occupations.

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
