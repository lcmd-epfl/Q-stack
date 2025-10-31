qstack.spahm.compute\_spahm
===========================

Functions
---------

get\_guess\_orbitals (mol, guess, xc='pbe', field=None, return\_ao\_dip=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the guess Hamiltonian orbitals

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe.
        field (numpy.array(3)): applied uniform electric field i.e. $\vec \nabla \phi$ in a.u.
        return_ao_dip (bool): if return computed AO dipole integrals

    Returns:
        1D numpy array containing the eigenvalues
        2D numpy array containing the eigenvectors of the guess Hamiltonian.
        (optional) 2D numpy array with the AO dipole integrals

ext\_field\_generator (mol, field)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Generator for Hext (i.e. applied uniform electiric field interaction) gradient

    Args:
        mol (pyscf Mole): pyscf Mole object.
        field (numpy.array(3)): applied uniform electric field i.e. $\vec \nabla \phi$ in a.u.

    Returns:
        func(int: iat): returns the derivative of Hext wrt the coordinates of atom iat, i.e. dHext/dr[iat]

get\_guess\_orbitals\_grad (mol, guess, field=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the guess Hamiltonian eigenvalues and their derivatives

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Tuple of methods used to compute the guess Hamiltonian and its eigenvalue derivatives. Output of get_guess_g
        field (numpy.array(3)): applied uniform electric field i.e. $\vec \nabla \phi$ in a.u.

    Returns:
        numpy 1d array (mol.nao,): eigenvalues
        numpy 3d ndarray (mol.nao,mol.natm,3): gradient of the eigenvalues in Eh/bohr
        numpy 2d ndarray (mol.nao,3): derivative of the eigenvalues wrt field in Eh/a.u.

get\_guess\_dm (mol, guess, xc='pbe', openshell=None, field=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the density matrix with the guess Hamiltonian.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe
        openshell (bool): . Defaults to None.

    Returns:
        A numpy ndarray containing the density matrix computed using the guess Hamiltonian.

get\_spahm\_representation (mol, guess\_in, xc='pbe', field=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the SPAHM representation.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Method used to obtain the guess Hamiltoninan.
        xc (str): Exchange-correlation functional. Defaults to pbe.
        field (numpy.array(3)): applied uniform electric field i.e. $\vec \nabla \phi$ in a.u.

    Returns:
        A numpy ndarray containing the SPAHM representation.

get\_spahm\_representation\_grad (mol, guess\_in, field=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the SPAHM representation and its gradient

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Method used to obtain the guess Hamiltoninan.
        field (numpy.array(3)): applied uniform electric field i.e. $\vec \nabla \phi$ in a.u.

    Returns:
        numpy 1d array (occ,): the SPAHM representation (Eh).
        numpy 3d array (occ,mol.natm,3): gradient of the representation (Eh/bohr)
        numpy 2d array (occ,3): gradient of the representation wrt electric field (Eh/a.u.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
