from pyscf import scf, grad
from qstack.spahm.guesses import solveF, get_guess, get_occ, get_dm, eigenvalue_grad, get_guess_g

def get_guess_orbitals(mol, guess, xc="pbe"):
    """ Compute the guess Hamiltonian.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (funct): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe.

    Returns:
        A 1D numpy array containing the eigenvalues and a 2D numpy array containing the eigenvectors of the guess Hamiltonian.
    """
    if guess == 'huckel':
        e,v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
        fock = guess(mol, xc)
        e,v = solveF(mol, fock)
    return e,v

def get_guess_orbitals_grad(mol, guess):
    e, c = get_guess_orbitals(mol, guess[0])
    mf = grad.rhf.Gradients(scf.RHF(mol))
    s1 = mf.get_ovlp(mol)
    h1 = guess[1](mf)
    return eigenvalue_grad(mol, e, c, s1, h1)

def get_guess_dm(mol, guess, xc="pbe", openshell=None):
    """ Compute the density matrix with the guess Hamiltonian.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (funct): Method used to compute the guess Hamiltonian. Output of get_guess.
        xc (str): Exchange-correlation functional. Defaults to pbe
        openshell (bool): . Defaults to None.

    Returns:
        A numpy ndarray containing the density matrix computed using the guess Hamiltonian.
    """
    e,v = get_guess_orbitals(mol, guess, xc)
    return get_dm(v, mol.nelec, mol.spin if mol.spin>0 or not openshell is None else None)

def get_spahm_representation(mol, guess_in, xc="pbe"):
    """ Compute the SPAHM representation.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Method used to obtain the guess Hamiltoninan.
        xc (str): Exchange-correlation functional. Defaults to pbe.

    Returns:
        A numpy ndarray containing the SPAHM representation.
    """
    guess = get_guess(guess_in)
    e, v  = get_guess_orbitals(mol, guess, xc)
    e1    = get_occ(e, mol.nelec, mol.spin)
    return e1

def get_spahm_representation_grad(mol, guess_in):
    guess = get_guess_g(guess_in)
    agrad = get_guess_orbitals_grad(mol, guess)
    return get_occ(agrad, mol.nelec, mol.spin)
