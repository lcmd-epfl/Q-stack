qstack.tools
============

Functions
---------

\_orca2gpr\_idx (mol)
~~~~~~~~~~~~~~~~~~~~~

::

    Given a molecule returns a list of reordered indices to tranform orca AO ordering into SA-GPR.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.

\_orca2gpr\_sign (mol)
~~~~~~~~~~~~~~~~~~~~~~

::

    Given a molecule returns a list of multipliers needed to tranform from orca AO.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of +1/-1 multipliers

\_pyscf2gpr\_idx (mol)
~~~~~~~~~~~~~~~~~~~~~~

::

    Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.

reorder\_ao (mol, vector, src='pyscf', dest='gpr')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Reorder the atomic orbitals from one convention to another.
    For example, src=pyscf dest=gpr reorders p-orbitals from +1,-1,0 (pyscf convention) to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        vector (numpy ndarray): vector or matrix
        src (string): current convention
        dest (string): convention to convert to (available: 'pyscf', 'gpr', ...

    Returns:
        A numpy ndarray with the reordered vector or matrix.

\_Rz (a)
~~~~~~~~

::

    Computes the rotation matrix around absolute z-axis.

    Args:
        a (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.

\_Ry (b)
~~~~~~~~

::

    Computes the rotation matrix around absolute y-axis.

    Args:
        b (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.

\_Rx (g)
~~~~~~~~

::

    Computes the rotation matrix around absolute x-axis.

    Args:
        g (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.

rotate\_euler (a, b, g, rad=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the rotation matrix given Euler angles.

    Args:
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool) : Wheter the angles are in radians or not.

    Returns:
        A 2D numpy ndarray with the rotation matrix.

unix\_time\_decorator (func)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

unix\_time\_decorator\_with\_tvalues (func)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

correct\_num\_threads ()
~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
