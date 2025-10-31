qstack.basis\_opt.basis\_tools
==============================

Functions
---------

energy\_mol (newbasis, moldata)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        numpy ndarray: Overlap matrix, 2-centers and 3-centers ERI matrices.

gradient\_mol (nexp, newbasis, moldata)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Args:
        nexp():
        newbasis():
        moldata(pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set

    Returns:

exp2basis (exponents, elements, basis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Argas:
        exponents():
        elements():
        basis():

    Returns:
        newbasis():

cut\_myelements (x, myelements, bf\_bounds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

printbasis (basis, f)
~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
