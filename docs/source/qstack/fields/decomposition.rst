qstack.fields.decomposition
===========================

Functions
---------

decompose (mol, dm, auxbasis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        dm (2D numpy array): Density matrix.
        auxbasis (string / pyscf basis dictionary): Atom-centered basis to decompose on.

    Returns:
        A copy of the pyscf Mole object with the auxbasis basis in a pyscf Mole object, and a 1D numpy array containing the decomposition coefficients.

get\_integrals (mol, auxmol)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        Three numpy ndarray containing: the overlap matrix, the 2-centers ERI matrix, and the 3-centers ERI matrix respectively.

get\_self\_repulsion (mol, dm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the Einstein summation of the Coulumb matrix and the density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Density matrix.

    Returns:
        A nummpy ndarray result of the Einstein summation of the J matrix and the Density matrix.

decomposition\_error (self\_repulsion, c, eri2c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the decomposition error.

    .. todo::
        Write the complete docstring

get\_coeff (dm, eri2c, eri3c, slices=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the density expansion coefficients.

    Args:
        dm (numpy ndarray): Density matrix.
        eri2c (numpy ndarray): 2-centers ERI matrix.
        eri3c (numpy ndarray): 3-centers ERI matrix.
        slices (optional numpy ndarray): assume that eri2c is bloc-diagonal, by giving the boundaries of said blocks

    Returns:
        A numpy ndarray containing the expansion coefficients of the density onto the auxiliary basis.

\_get\_inv\_metric (mol, metric, v)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Args:
      mol (pyscf Mole): pyscf Mole object.
      metric (str): unit, overlap or coulomb.
      v (numpy ndarray): Number of electrons decomposed into a vector.

correct\_N\_atomic (mol, N, c0, metric='u')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        N (int): Number of electrons. Defaults to None.
        c0 (1D numpy array): Decomposition coefficients.
        metric (str): .Defaults to 'u'.

    Returns:

    .. todo::
        Write the complete docstring.

correct\_N (mol, c0, N=None, mode='Lagrange', metric='u')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        c0 (1D numpy array): Decomposition coefficients.
        N (int): Number of electrons. Defaults to None.
        mode (str): Defaults to Lagrange.
        metric (str): Defaults to u.

    Returns:
        A numpy ndarray containing a set of expansion coefficients taking into account the correct total number of electrons.

    .. todo::
        Write the complete docstring.

number\_of\_electrons\_deco\_vec (mol, per\_atom=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    .. todo::
        Write the complete docstring.

number\_of\_electrons\_deco (auxmol, c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the number of electrons of a molecule given a set of expansion coefficients and a Mole object.

    Args:
        auxmol (pyscf Mole): pyscf mol object holding molecular structure, composition and the auxiliary basis set.
        c (numpy ndarray): expansion coefficients of the density onto the auxiliary basis.

    Returns:
        The number of electrons as an integer value.

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
