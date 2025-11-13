"""Initial guess Hamiltonian methods for SPAHM.

Implements various guess methods: Hcore, Hückel, GWH, SAD, SAP, LB2020.

Provides:
    - guesses_dict: Dictionary mapping guess names to functions.
"""

import warnings
import numpy as np
import scipy
from pyscf import dft, scf
from .LB2020guess import LB2020guess as LB20


def hcore(mol, *_):
    """Computes guess Hamiltonian from core contributions (kinetic + nuclear + ECP).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        *_: Unused positional arguments (for interface compatibility).

    Returns:
        numpy ndarray: 2D array containing the core Hamiltonian matrix in AO basis.
    """
    return scf.hf.get_hcore(mol)


def GWH(mol, *_):
    """Computes guess Hamiltonian using Generalized Wolfsberg-Helmholtz (GWH) method.

    Uses the formula: H_ij = 0.5 * K * (H_ii + H_jj) * S_ij with K = 1.75.

    Reference:
        M. Wolfsberg, L. Helmholtz,
        "The spectra and electronic structure of the tetrahedral ions MnO4-, CrO4--, and ClO4-",
        J. Chem. Phys. 20 837-843 (1952), doi:10.1063/1.1700580

    Args:
        mol (pyscf Mole): pyscf Mole object.
        *_: Unused positional arguments (for interface compatibility).

    Returns:
        numpy ndarray: 2D GWH Hamiltonian matrix in AO basis.
    """
    h = hcore(mol).diagonal()
    S = mol.intor_symmetric('int1e_ovlp')
    K = 1.75
    h_gwh = 0.5 * K * (h[:,None] + h[None,:]) * S
    h_gwh[np.diag_indices_from(h_gwh)] = h
    return h_gwh


def SAD(mol, xc):
    """Computes guess Hamiltonian using Superposition of Atomic Densities (SAD).

    Constructs the Fock matrix from atomic Hartree-Fock density matrices
    summed together as an initial guess for molecular calculations.

    References:
        J. Almlöf, K. Faegri Jr, K. Korsell,
        "Principles for a direct SCF approach to LICAO–MO ab-initio calculations",
        J. Comput. Chem. 3, 385–399 (1982), doi:10.1002/jcc.540030314

        L. Amat, R. Carbó-Dorca,
        "Use of promolecular ASA density functions as a general algorithm to obtain starting MO in SCF calculations",
        Int. J. Quantum Chem. 87, 59–67 (2001), doi:10.1002/qua.10068

        J. H. Van Lenthe, R. Zwaans, H. J. J. Van Dam and M. F. Guest,
        "Starting SCF calculations by superposition of atomic densities",
        J. Comput. Chem. 27, 926–932 (2006), doi:10.1002/jcc.20393

    Args:
        mol (pyscf Mole): pyscf Mole object.
        xc (str): Exchange-correlation functional.

    Returns:
        numpy ndarray: 2D Fock matrix in AO basis computed from SAD.

    Warns:
      RuntimeWarning: If alpha and beta effective potentials differ for the functional.
    """
    hc = hcore(mol)
    dm = scf.hf.init_guess_by_atom(mol)
    mf = dft.RKS(mol)
    mf.xc = xc
    vhf = mf.get_veff(dm=dm)
    if vhf.ndim == 2:
        fock = hc + vhf
    else:
        fock = hc + vhf[0]
        if not np.array_equal(vhf[0], vhf[1]):
          msg = f'The effective potential ({xc}) returned different alpha and beta matrix components from atomicHF DM'
          warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return fock


def SAP(mol, *_):
    """Computes guess Hamiltonian using Superposition of Atomic Potentials (SAP).

    Constructs initial Hamiltonian from kinetic energy plus summed atomic potentials.

    Reference:
        S. Lehtola,
        "Assessment of initial guesses for self-consistent field calculations.
         Superposition of atomic potentials: Simple yet efficient",
        J. Chem. Theory Comput. 15, 1593 (2019), doi:10.1021/acs.jctc.8b01089

    Args:
        mol (pyscf Mole): pyscf Mole object.
        *_: Unused positional arguments (for interface compatibility).

    Returns:
        numpy ndarray: 2D Hamiltonian matrix (T + V_SAP) in AO basis.
    """
    mf = dft.RKS(mol)
    vsap = mf.get_vsap()
    t = mol.intor_symmetric('int1e_kin')
    fock = t + vsap
    return fock


def LB(mol, *_):
    """Computes guess Hamiltonian using Laikov-Briling 2020 model with HF parameters.

    Reference:
        D. N. Laikov, K. R. Briling,
        "Atomic effective potentials for starting molecular electronic structure calculations",
        Theor. Chem. Acc. 139, 17 (2020), doi:10.1007/s00214-019-2521-3

    Args:
        mol (pyscf Mole): pyscf Mole object.
        *_: Unused positional arguments (for interface compatibility).

    Returns:
        numpy ndarray: 2D effective Hamiltonian matrix from LB2020 model in AO basis.
    """
    return LB20(parameters='HF').Heff(mol)


def LB_HFS(mol, *_):
    """Computes guess Hamiltonian using Laikov-Briling 2020 model with HFS parameters.

    Reference:
        D. N. Laikov, K. R. Briling,
        "Atomic effective potentials for starting molecular electronic structure calculations",
        Theor. Chem. Acc. 139, 17 (2020), doi:10.1007/s00214-019-2521-3

    Args:
        mol (pyscf Mole): pyscf Mole object.
        *_: Unused positional arguments (for interface compatibility).

    Returns:
        numpy ndarray: 2D effective Hamiltonian matrix from LB2020-HFS model in AO basis.
    """
    return LB20(parameters='HFS').Heff(mol)


def solveF(mol, fock):
    """Solves generalized eigenvalue problem FC = SCε for the Fock/Hamiltonian matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        fock (numpy ndarray): 2D Fock or Hamiltonian matrix in AO basis.

    Returns:
        tuple: (eigenvalues, eigenvectors) where:
        - eigenvalues: 1D array of orbital energies
        - eigenvectors: 2D array of MO coefficients (columns are MOs)
    """
    s1e = mol.intor_symmetric('int1e_ovlp')
    return scipy.linalg.eigh(fock, s1e)


def get_guess(arg):
    """Returns guess Hamiltonian function by name.

    Args:
        arg (str): Guess method name. Available options:
            - 'core': Core Hamiltonian (H_core).
            - 'sad': Superposition of Atomic Densities.
            - 'sap': Superposition of Atomic Potentials.
            - 'gwh': Generalized Wolfsberg-Helmholtz.
            - 'lb': Laikov-Briling 2020 (HF parameters).
            - 'lb-hfs': Laikov-Briling 2020 (HFS parameters).
            - 'huckel': Extended Hückel method.

    Returns:
        callable: Guess Hamiltonian function with signature f(mol, xc) -> numpy.ndarray.

    Raises:
        RuntimeError: If the specified guess method is not available.
    """
    arg = arg.lower()
    if arg not in guesses_dict:
        raise RuntimeError(f'Unknown guess. Available guesses: {list(guesses_dict.keys())}')
    return guesses_dict[arg]


def check_nelec(nelec, nao):
    """Validates that the number of electrons can be accommodated by available orbitals.

    Args:
        nelec (tuple or int): Number of electrons (alpha, beta) or total.
        nao (int): Number of atomic orbitals.

    Raises:
        RuntimeError: If there are more electrons than available orbitals.

    Warns:
        RuntimeWarning: If all orbitals are filled (complete shell warning).
    """
    if np.any(np.array(nelec) > nao):
        raise RuntimeError(f'Too many electrons ({nelec}) for {nao} orbitals')
    elif np.any(np.array(nelec) == nao):
        msg = f'{nelec} electrons for {nao} orbitals. Is the input intended to have a complete shell?'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def get_occ(e, nelec, spin):
    """Extracts occupied orbital eigenvalues/energies.

    Args:
        e (numpy ndarray): Full array of orbital eigenvalues (1D)
            or possibly arrays of larger dimensionality.
        nelec (tuple): Number of (alpha, beta) electrons.
        spin (int or None): Spin multiplicity. If None, assumes closed-shell.

    Returns:
        numpy ndarray: Occupied eigenvalues. Shape depends on spin:
        - Closed-shell (spin=None): 1D array of occupied eigenvalues
        - Open-shell: 2D array (2, nocc) for alpha and beta separately
    """
    check_nelec(nelec, e.shape[0])
    if spin is None:
        nocc = nelec[0]
        return e[:nocc,...]
    else:
        nocc = nelec
        e1 = np.zeros((2, nocc[0], *e.shape[1:]))
        e1[0,:nocc[0],...] = e[:nocc[0],...]
        e1[1,:nocc[1],...] = e[:nocc[1],...]
        return e1


def get_dm(v, nelec, spin):
    """Constructs density matrix from occupied molecular orbitals.

    Args:
        v (numpy ndarray): 2D array of MO coefficients (eigenvectors), columns are MOs.
        nelec (tuple): Number of (alpha, beta) electrons.
        spin (int or None): Spin multiplicity. If None, assumes closed-shell (RHF).

    Returns:
        numpy ndarray: Density matrix in AO basis.
        - Closed-shell: 2D array (nao, nao)
        - Open-shell: 3D array (2, nao, nao) for alpha and beta
    """
    check_nelec(nelec, len(v))
    if spin is None:
        nocc = nelec[0]
        dm = v[:,:nocc] @ v[:,:nocc].T
        return 2.0*dm
    else:
        nocc = nelec
        dm0 = v[:,:nocc[0]] @ v[:,:nocc[0]].T
        dm1 = v[:,:nocc[1]] @ v[:,:nocc[1]].T
        return np.array((dm0,dm1))


def hcore_grad(mf):
    """Returns core Hamiltonian gradient generator function.

    Args:
        mf: PySCF mean-field object.

    Returns:
        callable: Function that returns core Hamiltonian gradient for a given atom.
    """
    return mf.hcore_generator(mf.mol)


def LB_grad(mf):
    """Returns Laikov-Briling Hamiltonian gradient generator function.

    Combines core Hamiltonian gradient with LB2020 model gradient.

    Args:
        mf: Mean-field object with hcore_generator method.

    Returns:
        callable: Function that returns total Hamiltonian gradient for a given atom.
    """
    hcore_grad = mf.hcore_generator(mf.mol)
    HLB_grad   = LB20().HLB20_generator(mf.mol)
    def H_grad(iat):
        return hcore_grad(iat) + HLB_grad(iat)
    return H_grad


def get_guess_g(arg):
    """Returns both guess Hamiltonian function and its gradient generator.

    Args:
        arg (str): Guess method name. Available: 'core', 'lb'.

    Returns:
        tuple: (hamiltonian_function, gradient_function) pair.

    Raises:
        RuntimeError: If the specified guess method is not available for gradients.
    """
    arg = arg.lower()
    guesses = {'core':(hcore, hcore_grad), 'lb':(LB, LB_grad)}
    if arg not in guesses:
        raise RuntimeError(f'Unknown guess. Available guesses: {list(guesses.keys())}')
    return guesses[arg]


def eigenvalue_grad(mol, e, c, s1, h1):
    """Computes nuclear gradients of orbital eigenvalues from generalized eigenvalue problem HC = eSC.

    Uses the Hellmann-Feynman theorem for eigenvalue derivatives.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        e (numpy ndarray): 1D array (nao,) of orbital eigenvalues.
        c (numpy ndarray): 2D array (nao, nao) of MO coefficients (eigenvectors).
        s1 (numpy ndarray): 3D array (3, nao, nao) - gradient of overlap matrix.
        h1 (callable): Function returning dH/dr[iat] - Hamiltonian gradient for atom iat.

    Returns:
        numpy ndarray: 3D array (nao, natm, 3) of eigenvalue gradients in Eh/bohr.
    """
    de_dr = np.zeros((mol.nao, mol.natm, 3))
    aoslices = mol.aoslice_by_atom()[:,2:]
    for iat in range(mol.natm):
        dH_dr = h1(iat)
        p0, p1 = aoslices[iat]
        Hcomp = np.einsum('pi,aqp,qi->ia', c, dH_dr, c)
        Scomp = 2.0 * np.einsum('pi,aqp,qi->ia', c, s1[:,p0:p1], c[p0:p1])
        de_dr[:,iat,:] = Hcomp - Scomp * e[:,None]
    return de_dr


guesses_dict = {'core':hcore, 'sad':SAD, 'sap':SAP, 'gwh':GWH, 'lb':LB, 'huckel':'huckel', 'lb-hfs':LB_HFS}
