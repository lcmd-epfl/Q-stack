qstack.spahm.guesses
====================

Functions
---------

hcore (mol, \*\_)
~~~~~~~~~~~~~~~~~

::

    Uses the core potential (kin + nuc + ecp) to compute the guess Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

GWH (mol, \*\_)
~~~~~~~~~~~~~~~

::

    Uses the generalized Wolfsberg-Helmholtz to compute the guess Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

SAD (mol, func)
~~~~~~~~~~~~~~~

::

    Uses the superposition of atomic densities to compute the guess Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.
      func (str): Exchange-correlation functional.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

SAP (mol, \*\_)
~~~~~~~~~~~~~~~

::

    Uses the superposition of atomic potentials to compute the guess Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

LB (mol, \*\_)
~~~~~~~~~~~~~~

::

    Uses the Laikov-Briling model with HF-based parameters to compute the guess Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

LB\_HFS (mol, \*\_)
~~~~~~~~~~~~~~~~~~~

::

    Laikov-Briling using HFS-based parameters

    Args:
      mol (pyscf Mole): pyscf Mole object.

    Returns:
      A numpy ndarray containing the computed approximate Hamiltonian.

solveF (mol, fock)
~~~~~~~~~~~~~~~~~~

::

    Computes the eigenvalues and eigenvectors corresponding to the given Hamiltonian.

    Args:
      mol (pyscf Mole): pyscf Mole object.
      fock (numpy ndarray): Approximate Hamiltonian.

get\_guess (arg)
~~~~~~~~~~~~~~~~

::

    Returns the function of the method selected to compute the approximate hamiltoninan

    Args:
      arg (str): Approximate Hamiltonian

    Returns:
      The function of the selected method.

check\_nelec (nelec, nao)
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Checks if there is enough orbitals
    for the electrons

get\_occ (e, nelec, spin)
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Returns the occupied subset of e

    Args:
      e (numpy ndarray): Energy eigenvalues.
      nelec(tuple): Number of alpha and beta electrons.
      spin(int): Spin.

    Returns:
      A numpy ndarray containing the occupied eigenvalues.

get\_dm (v, nelec, spin)
~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the density matrix.

    Args:
      v (numpy ndarray): Eigenvectors of a previously solve Hamiltoinan.
      nelec(tuple): Number of alpha and beta electrons.
      spin(int): Spin.

    Return:
      A numpy ndarray containing the density matrix computed using the guess Hamiltonian.

hcore\_grad (mf)
~~~~~~~~~~~~~~~~

(No docstring.)

LB\_grad (mf)
~~~~~~~~~~~~~

(No docstring.)

get\_guess\_g (arg)
~~~~~~~~~~~~~~~~~~~

(No docstring.)

eigenvalue\_grad (mol, e, c, s1, h1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute gradients of eigenvalues found from HC=eSC

    Args:
        mol (pyscf Mole): pyscf Mole object
        e (numpy 1d ndarray, mol.nao): eigenvalues
        c (numpy 2d ndarray, mol.nao*mol.nao): eigenvectors
        s1 (numpy 3d ndarray, 3*mol.nao*mol.nao): compact gradient of the overlap matrix [-(nabla \|\)]
        h1 (func(int: iat)): returns the derivative of H wrt the coordinates of atom iat, i.e. dH/dr[iat]

    Returns:
        numpy 3d ndarray, mol.nao*mol.natm*3: gradient of the eigenvalues in Eh/bohr

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
