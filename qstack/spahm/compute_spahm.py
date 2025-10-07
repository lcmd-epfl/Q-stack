import numpy as np
from pyscf import scf, grad
from .guesses import solveF, get_guess, get_occ, get_dm, eigenvalue_grad, get_guess_g


def get_guess_orbitals(mol, guess, xc="pbe", field=None, return_ao_dip=False):
    """ Compute the guess Hamiltonian orbitals

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe.
        field (numpy.array(3)): applied uniform electric field i.e. $\\vec \\nabla \\phi$ in a.u.
        return_ao_dip (bool): if return computed AO dipole integrals

    Returns:
        1D numpy array containing the eigenvalues
        2D numpy array containing the eigenvectors of the guess Hamiltonian.
        (optional) 2D numpy array with the AO dipole integrals
    """
    if guess == 'huckel':
        if field is not None:
            raise NotImplementedError
        e, v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
        fock = guess(mol, xc)
        if field is not None:
            with mol.with_common_orig((0,0,0)):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
            fock += np.einsum('xij,x->ij', ao_dip, field)
        else:
            ao_dip = None
        e, v = solveF(mol, fock)
    if return_ao_dip:
        return e, v, ao_dip
    else:
        return e, v


def ext_field_generator(mol, field):
    """ Generator for Hext (i.e. applied uniform electiric field interaction) gradient

    Args:
        mol (pyscf Mole): pyscf Mole object.
        field (numpy.array(3)): applied uniform electric field i.e. $\\vec \\nabla \\phi$ in a.u.

    Returns:
        func(int: iat): returns the derivative of Hext wrt the coordinates of atom iat, i.e. dHext/dr[iat]
    """

    shls_slice = (0, mol.nbas, 0, mol.nbas)
    with mol.with_common_orig((0,0,0)):
        int1e_irp = mol.intor('int1e_irp', shls_slice=shls_slice).reshape(3, 3, mol.nao, mol.nao) # ( | rc nabla | )
    aoslices = mol.aoslice_by_atom()[:,2:]
    if field is None:
        field = (0,0,0)
    def field_deriv(iat):
        p0, p1 = aoslices[iat]
        dmu_dr = np.zeros_like(int1e_irp)  # dim(mu)×dim(r)×nao×nao
        dmu_dr[:,:,p0:p1,:] -= int1e_irp[:,:,:,p0:p1].transpose((0,1,3,2))  # TODO not sure why minus
        dmu_dr[:,:,:,p0:p1] -= int1e_irp[:,:,:,p0:p1]  # TODO check/fix E definition
        dhext_dr = np.einsum('x,xypq->ypq', field, dmu_dr)
        return dhext_dr
    return field_deriv


def get_guess_orbitals_grad(mol, guess, field=None):
    """ Compute the guess Hamiltonian eigenvalues and their derivatives

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Tuple of methods used to compute the guess Hamiltonian and its eigenvalue derivatives. Output of get_guess_g
        field (numpy.array(3)): applied uniform electric field i.e. $\\vec \\nabla \\phi$ in a.u.

    Returns:
        numpy 1d array (mol.nao,): eigenvalues
        numpy 3d ndarray (mol.nao,mol.natm,3): gradient of the eigenvalues in Eh/bohr
        numpy 2d ndarray (mol.nao,3): derivative of the eigenvalues wrt field in Eh/a.u.
    """

    e, c, ao_dip = get_guess_orbitals(mol, guess[0], field=field, return_ao_dip=True)
    mf = grad.rhf.Gradients(scf.RHF(mol))
    s1 = mf.get_ovlp(mol)
    h0 = guess[1](mf)

    if field is None:
        h1 = h0
        de_dfield = None
    else:
        hext = ext_field_generator(mf.mol, field)
        h1 = lambda iat: h0(iat) + hext(iat)
        de_dfield = np.einsum('pi,aqp,qi->ia', c, ao_dip, c)
    return e, eigenvalue_grad(mol, e, c, s1, h1), de_dfield


def get_guess_dm(mol, guess, xc="pbe", openshell=None, field=None):
    """ Compute the density matrix with the guess Hamiltonian.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (func): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe
        openshell (bool): . Defaults to None.

    Returns:
        A numpy ndarray containing the density matrix computed using the guess Hamiltonian.
    """
    e,v = get_guess_orbitals(mol, guess, xc, field=field)
    return get_dm(v, mol.nelec, mol.spin if mol.spin>0 or not openshell is None else None)


def get_spahm_representation(mol, guess_in, xc="pbe", field=None):
    """ Compute the SPAHM representation.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Method used to obtain the guess Hamiltoninan.
        xc (str): Exchange-correlation functional. Defaults to pbe.
        field (numpy.array(3)): applied uniform electric field i.e. $\\vec \\nabla \\phi$ in a.u.

    Returns:
        A numpy ndarray containing the SPAHM representation.
    """
    guess = get_guess(guess_in)
    e, v  = get_guess_orbitals(mol, guess, xc, field=field)
    e1    = get_occ(e, mol.nelec, mol.spin)
    return e1


def get_spahm_representation_grad(mol, guess_in, field=None):
    """ Compute the SPAHM representation and its gradient

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Method used to obtain the guess Hamiltoninan.
        field (numpy.array(3)): applied uniform electric field i.e. $\\vec \\nabla \\phi$ in a.u.

    Returns:
        numpy 1d array (occ,): the SPAHM representation (Eh).
        numpy 3d array (occ,mol.natm,3): gradient of the representation (Eh/bohr)
        numpy 2d array (occ,3): gradient of the representation wrt electric field (Eh/a.u.)
    """
    guess = get_guess_g(guess_in)
    e, agrad, fgrad = get_guess_orbitals_grad(mol, guess, field=field)

    return (get_occ(e, mol.nelec, mol.spin),
            get_occ(agrad, mol.nelec, mol.spin),
            get_occ(fgrad, mol.nelec, mol.spin) if fgrad is not None else None)
