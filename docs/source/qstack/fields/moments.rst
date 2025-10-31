qstack.fields.moments
=====================

Functions
---------

first (mol, rho)
~~~~~~~~~~~~~~~~

::

    Computes the transition dipole moments.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        rho (numpy ndarray): Density Matrix (trnasition if given ) or fitting coefficnts for the same matrix.

    Returns:
        A numpy ndarray with the transition dipole moments. If rho is a 1D matrix, returns the Decomposed/predicted transition dipole moments; if rho is a 2D matrix, returns the ab initio transition dipole moments.

r\_dm (mol, dm)
~~~~~~~~~~~~~~~

::

    .. todo::
        write docstring.

r\_c (mol, rho)
~~~~~~~~~~~~~~~

::

    .. todo::
        Write docstring, and include uncontracted basis in code and verify formulas

r2\_c (rho, mol)
~~~~~~~~~~~~~~~~

::

    Compute the zeroth ( :math:`<1>` ), first ( :math:`<r>` ), and second ( :math:`<r^{2}>`) moments of electron density distribution.

    .. math::

        <1> = \int \rho d r
        \quad
        ;
        \quad
        <r> = \int \hat{r} \rho d r
        \quad
        ;
        \quad
        <r^{2}> = \int \hat{r}^{2} \rho d r

    Args:
        mol (scipy Mole): scipy Mole object.

    Returns:
        The zeroth, first, and second moments of electron density distribution.

    .. todo::
        Include uncontracted basis in code and verify formulas

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
