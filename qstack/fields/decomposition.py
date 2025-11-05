import numpy as np
import scipy
from pyscf import scf
from qstack import compound
from . import moments

def decompose(mol, dm, auxbasis):
    """Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        dm (2D numpy array): Density matrix.
        auxbasis (string / pyscf basis dictionary): Atom-centered basis to decompose on.

    Returns:
        A copy of the pyscf Mole object with the auxbasis basis in a pyscf Mole object, and a 1D numpy array containing the decomposition coefficients.
    """

    auxmol = compound.make_auxmol(mol, auxbasis)
    _S, eri2c, eri3c = get_integrals(mol, auxmol)
    c = get_coeff(dm, eri2c, eri3c)
    return auxmol, c

def get_integrals(mol, auxmol):
    """Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        Three numpy ndarray containing: the overlap matrix, the 2-centers ERI matrix, and the 3-centers ERI matrix respectively.
    """

    # Get overlap integral in the auxiliary basis
    S = auxmol.intor('int1e_ovlp_sph')

    # Concatenate standard and auxiliary basis set into a pmol object
    pmol = mol + auxmol

    # Compute 2- and 3-centers ERI integrals using the concatenated mol object
    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

    return S, eri2c, eri3c

def get_self_repulsion(mol, dm):
    """Computes the Einstein summation of the Coulumb matrix and the density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Density matrix.

    Returns:
        A nummpy ndarray result of the Einstein summation of the J matrix and the Density matrix.
    """

    try:
        j, _k = mol.get_jk()
    except AttributeError:
        j, _k = scf.hf.get_jk(mol, dm)
    return np.einsum('ij,ij', j, dm)

def decomposition_error(self_repulsion, c, eri2c):
    """Computes the decomposition error for density fitting.

    Args:
        self_repulsion (float): Self-repulsion energy from the original density matrix.
        c (numpy ndarray): 1D array of density expansion coefficients.
        eri2c (numpy ndarray): 2D array of 2-center electron repulsion integrals.

    Returns:
        float: The decomposition error.
    """
    return self_repulsion - c @ eri2c @ c

def get_coeff(dm, eri2c, eri3c, slices=None):
    """Computes the density expansion coefficients.

    Args:
        dm (numpy ndarray): Density matrix.
        eri2c (numpy ndarray): 2-centers ERI matrix.
        eri3c (numpy ndarray): 3-centers ERI matrix.
        slices (optional numpy ndarray): assume that eri2c is bloc-diagonal, by giving the boundaries of said blocks

    Returns:
        A numpy ndarray containing the expansion coefficients of the density onto the auxiliary basis.
    """

    # Compute the projection of the density onto auxiliary basis using a Coulomb metric
    projection = np.einsum('ijp,ij->p', eri3c, dm)

    # Solve Jc = projection to get the coefficients
    if slices is not None:
        if not (slices.ndim==2 and slices.shape[0]>0 and slices.shape[1]==2) or\
           not (slices[0,0] == 0 and slices[-1,1] == projection.shape[0]):
            raise RuntimeError(f"Wrong argument {slices=}")

        c = np.empty_like(projection)
        for s0,s1 in slices:
            c[s0:s1] = scipy.linalg.solve(eri2c[s0:s1,s0:s1], projection[s0:s1], assume_a='pos')
    else:
        c = scipy.linalg.solve(eri2c, projection, assume_a='pos')

    return c

def _get_inv_metric(mol, metric, v):
    """Computes the inverse metric applied to a vector.

    Args:
      mol (pyscf Mole): pyscf Mole object.
      metric (str or numpy ndarray): Metric type ('unit', 'overlap', 'coulomb') or a metric matrix.
      v (numpy ndarray): Vector to apply the inverse metric to.

    Returns:
      numpy ndarray: Result of applying the inverse metric to the input vector.
    """
    if isinstance(metric, str):
        metric = metric.lower()
        if metric in ['u', 'unit', '1']:
            return v
        elif metric in ['s', 'overlap', 'ovlp']:
            O = mol.intor('int1e_ovlp_sph')
        elif metric in ['j', 'coulomb']:
            O = mol.intor('int2c2e_sph')
    else:
        O = metric
    return scipy.linalg.solve(O, v, assume_a='pos')


def correct_N_atomic(mol, N, c0, metric='u'):
    """Corrects decomposition coefficients to match the target electron count per atom.

    Uses Lagrange multipliers to enforce the correct number of electrons per atom
    while minimizing changes to the decomposition coefficients.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        N (numpy ndarray): Target number of electrons per atom.
        c0 (numpy ndarray): 1D array of initial decomposition coefficients.
        metric (str): Metric type for correction ('u' for unit, 's' for overlap, 'j' for coulomb). Defaults to 'u'.

    Returns:
        numpy ndarray: Corrected decomposition coefficients (1D array).
    """

    Q   = moments.r2_c(mol, None, moments=[0], per_atom=True)[0]
    N0  = c0 @ Q
    O1q = _get_inv_metric(mol, metric, Q)
    la  = scipy.linalg.solve(Q.T @ O1q, N-N0)
    c   = c0 + np.einsum('pq,q->p', O1q, la)
    return c


def correct_N(mol, c0, N=None, mode='Lagrange', metric='u'):
    """Corrects decomposition coefficients to match the target total electron count.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        c0 (numpy ndarray): 1D array of initial decomposition coefficients.
        N (int, optional): Target number of electrons. If None, uses mol.nelectron. Defaults to None.
        mode (str): Correction method ('scale' or 'lagrange'). Defaults to 'Lagrange'.
        metric (str): Metric type for Lagrange correction ('u', 's', or 'j'). Defaults to 'u'.

    Returns:
        numpy ndarray: Corrected decomposition coefficients (1D array).
    """

    mode = mode.lower()
    q = moments.r2_c(mol, None, moments=[0])
    N0 = c0 @ q

    if N is None:
        N = mol.nelectron

    if mode == 'scale':
        c = c0 * N/N0

    elif mode == 'lagrange' :
        O1q = _get_inv_metric(mol, metric, q)
        la  = (N - N0) / (q @ O1q)
        c   = c0 + la * O1q
    return c
