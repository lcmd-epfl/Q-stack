import numpy as np
from pyscf import scf, grad
from .guesses import solveF, get_guess, get_occ, get_dm, eigenvalue_grad, get_guess_g


def get_guess_orbitals(mol, guess, xc="pbe", field=None, return_ao_dip=False):
    """Computes molecular orbitals from a guess Hamiltonian with optional electric field.

    Solves the generalized eigenvalue problem for the guess Hamiltonian, optionally
    including an external uniform electric field perturbation.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (callable or str): Guess Hamiltonian method function (from get_guess) or 'huckel'.
        xc (str): Exchange-correlation functional name. Defaults to 'pbe'.
        field (numpy ndarray, optional): 3-component uniform electric field vector (∇φ) in atomic units.
            Defaults to None.
        return_ao_dip (bool): If True, also returns AO dipole integrals. Defaults to False.

    Returns:
        tuple: Depending on return_ao_dip:
            - If False: (e, v) where:
                - e (numpy ndarray): 1D array of orbital eigenvalues
                - v (numpy ndarray): 2D array (nao, nao) of MO coefficients
            - If True: (e, v, ao_dip) where ao_dip is 3D array (3, nao, nao) of AO dipole integrals

    Raises:
        NotImplementedError: If field is specified with Hückel guess.
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
    """Generates external electric field Hamiltonian gradient function.

    Creates a function that computes derivatives of the external field interaction
    Hamiltonian (H_ext) with respect to nuclear coordinates for each atom.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        field (numpy ndarray or None): 3-component uniform electric field vector (∇φ) in atomic units.
            If None, treated as zero field.

    Returns:
        callable: Function field_deriv(iat) that takes atom index and returns
            3D array (3, nao, nao) of dH_ext/dr[iat] - external field Hamiltonian
            gradient for atom iat.
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
    """Computes guess Hamiltonian eigenvalues and their nuclear/field gradients.

    Calculates orbital energies and their derivatives with respect to both nuclear
    coordinates (for geometry optimization/force calculations) and electric field
    (for response properties).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (tuple): Pair (hamiltonian_func, gradient_func) from get_guess_g().
        field (numpy ndarray, optional): 3-component uniform electric field (∇φ) in atomic units.
            Defaults to None.

    Returns:
        tuple: (e, de_dr, de_dfield) where:
            - e (numpy ndarray): 1D array (nao,) of orbital eigenvalues in Eh
            - de_dr (numpy ndarray): 3D array (nao, natm, 3) of eigenvalue gradients in Eh/bohr
            - de_dfield (numpy ndarray or None): 2D array (nao, 3) of eigenvalue derivatives
              w.r.t. electric field in Eh/a.u., or None if field is None
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
    """Computes density matrix from guess Hamiltonian orbitals.

    Constructs the density matrix by occupying the lowest energy orbitals from
    the guess Hamiltonian according to the aufbau principle.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess (callable): Guess Hamiltonian method function from get_guess().
        xc (str): Exchange-correlation functional name. Defaults to 'pbe'.
        openshell (bool, optional): If True, forces open-shell treatment even for closed-shell systems.
            Defaults to None.
        field (numpy ndarray, optional): 3-component uniform electric field (∇φ) in atomic units.
            Defaults to None.

    Returns:
        numpy ndarray: Density matrix in AO basis.
            - Closed-shell: 2D array (nao, nao)
            - Open-shell: 3D array (2, nao, nao) for alpha and beta separately
    """
    _e, v = get_guess_orbitals(mol, guess, xc, field=field)
    return get_dm(v, mol.nelec, mol.spin if mol.spin>0 or openshell is not None else None)


def get_spahm_representation(mol, guess_in, xc="pbe", field=None):
    """Computes the SPAHM (Single-Point Approximate Hamiltonian Method) molecular representation.

    The SPAHM representation consists of the occupied orbital energies from a guess
    Hamiltonian, providing a fast quantum-mechanical descriptor for machine learning.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Guess method name (e.g., 'LB', 'SAD', 'core', 'GWH').
        xc (str): Exchange-correlation functional name. Defaults to 'pbe'.
        field (numpy ndarray, optional): 3-component uniform electric field (∇φ) in atomic units.
            Defaults to None.

    Returns:
        numpy ndarray: SPAHM representation consisting of occupied orbital eigenvalues.
            - Closed-shell: 1D array of shape (n_occupied,) in Eh
            - Open-shell: 2D array of shape (2, n_alpha/n_beta) for alpha and beta orbitals
    """
    guess = get_guess(guess_in)
    e, _v = get_guess_orbitals(mol, guess, xc, field=field)
    e1    = get_occ(e, mol.nelec, mol.spin)
    return e1


def get_spahm_representation_grad(mol, guess_in, field=None):
    """Computes SPAHM representation and its nuclear/field gradients for force/response calculations.

    Calculates the SPAHM descriptor (occupied orbital energies) along with derivatives
    needed for molecular dynamics, geometry optimization, and response properties.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        guess_in (str): Guess method name with gradient support ('core' or 'lb').
        field (numpy ndarray, optional): 3-component uniform electric field (∇φ) in atomic units.
            Defaults to None.

    Returns:
        tuple: (spahm, spahm_grad, spahm_field_grad) where:
            - spahm (numpy ndarray): SPAHM representation - occupied orbital energies in Eh.
              Shape: (n_occ,) for closed-shell or (2, n_alpha/n_beta) for open-shell
            - spahm_grad (numpy ndarray): Nuclear gradients of SPAHM in Eh/bohr.
              Shape: (n_occ, natm, 3) or (2, n_alpha/n_beta, natm, 3)
            - spahm_field_grad (numpy ndarray or None): Electric field gradients in Eh/a.u.
              Shape: (n_occ, 3) or (2, n_alpha/n_beta, 3), or None if field is None
    """
    guess = get_guess_g(guess_in)
    e, agrad, fgrad = get_guess_orbitals_grad(mol, guess, field=field)

    return (get_occ(e, mol.nelec, mol.spin),
            get_occ(agrad, mol.nelec, mol.spin),
            get_occ(fgrad, mol.nelec, mol.spin) if fgrad is not None else None)
