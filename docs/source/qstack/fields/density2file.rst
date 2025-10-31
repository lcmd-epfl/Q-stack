qstack.fields.density2file
==========================

Functions
---------

coeffs\_to\_cube (mol, coeffs, cubename, nx=80, ny=80, nz=80, resolution=0.1, margin=3.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Saves the density in a cube file.

    Args:
        mol (pyscf Mole): pyscf Mole.
        coeffs (numpy ndarray): Expansion coefficients.
        cubename (str): Name of the cubo file.

    Returns:
        A new or overwrited file named <cubename>.cube

coeffs\_to\_molden (mol, coeffs, moldenname)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Saves the density in a molden file.

    Args:
        mol (pyscf Mole): pyscf Mole.
        coeffs (numpy ndarray): Expansion coefficients.
        moldenname (str): File name of the molden file.

    Returns:
        A new or overwrited file named <moldenname>.molden

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
