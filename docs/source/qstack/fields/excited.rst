qstack.fields.excited
=====================

Functions
---------

get\_cis (mf, nstates)
~~~~~~~~~~~~~~~~~~~~~~

::

    .. todo::
        Write the complete docstring.

get\_cis\_tdm (td)
~~~~~~~~~~~~~~~~~~

::

    .. todo::
        Write the complete docstring.

get\_holepart (mol, x, coeff)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the hole and particle density matrices (atomic orbital basis) of selected states.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x (numpy ndarray): Response vector (nstates×occ×virt) normalized to 1.
        coeff (numpy ndarray): Ground-state molecular orbital vectors.

    Returns:
        Two numpy ndarrays containing the hole density matrices and the particle density matrices respectively.

get\_transition\_dm (mol, x\_mo, coeff)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the Transition Density Matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x_mo (numpy ndarray): Response vector (nstates×occ×virt) normalized to 1.
        coeff (numpy ndarray): Ground-state molecular orbital vectors.

    Returns:
        A numpy ndarray containing the Transition Density Matrix.

exciton\_properties\_c (mol, hole, part)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the decomposed/predicted hole-particle distance, the hole size and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        Three floats: the hole-particle distance, the hole size, and the particle size respectively.

exciton\_properties\_dm (mol, hole, part)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the ab initio hole-particle distance, the hole size and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        Three floats: the hole-particle distance, the hole size, and the particle size respectively.

exciton\_properties (mol, hole, part)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the ab initio or decomposed/predicted hole-particle distance, the hole size and the particle size according to the number of dimensions of the density matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        The hole-particle distance, the hole size, and the particle size as floats.

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
