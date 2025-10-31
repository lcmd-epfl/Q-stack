qstack.fields.dm
================

Functions
---------

get\_converged\_dm (mol, xc, verbose=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Performs restricted SCF and returns density matrix, given pyscf mol object and an XC density functional.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        xc (str): Exchange-correlation functional.
        verbose (bool): If print more info

    Returns:
        A numpy ndarray containing the density matrix in AO-basis.

make\_grid\_for\_rho (mol, grid\_level=3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Generates a grid of real space coordinates and weights for integration.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        grid_level (int): Controls the number of radial and angular points.

    Returns:
        pyscf Grid object.

sphericalize\_density\_matrix (mol, dm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Sphericalize the density matrix in the sense of an integral over all possible rotations.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (2D numpy array): Density matrix in AO-basis.

    Returns:
        A numpy ndarray with the sphericalized density matrix.

get\_converged\_mf (mol, func, dm0=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    .. todo::
        Write the complete docstring, and merge with get_converged_dm()

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
