"""Density matrix decomposition onto auxiliary basis sets."""

import numpy as np
import scipy
from pyscf import scf, gto, dft
from qstack import compound
from . import moments
from .dm import make_grid_for_rho


def decompose(mol, dm, auxbasis):
    """Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        dm (2D numpy array): Density matrix.
        auxbasis (string / pyscf basis dictionary): Atom-centered basis to decompose on.

    Returns:
        Tuple containing:
        - copy of the pyscf Mole object with the auxbasis basis in a pyscf Mole object,
        - 1D numpy array containing the decomposition coefficients.
    """
    auxmol = compound.make_auxmol(mol, auxbasis)
    _S, eri2c, eri3c = get_integrals(mol, auxmol)
    c = get_coeff(dm, eri2c, eri3c)
    return auxmol, c


def get_integrals(mol, auxmol):
    """Compute overlap integrals and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object of the same molecule with an auxiliary basis set.

    Returns:
        Tuple of three numpy ndarray containing:
        - overlap matrix (auxmol.nao,auxmol.nao) for the auxiliary basis,
        - 2-centers ERI matrix (auxmol.nao,auxmol.nao) for the auxiliary basis,
        - 3-centers ERI matrix (mol.nao,mol.nao,auxmol.nao) between AO and auxiliary basis.
    """
    S = auxmol.intor('int1e_ovlp_sph')
    pmol = mol + auxmol
    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    return S, eri2c, eri3c


def get_self_repulsion(mol_or_mf, dm):
    r"""Compute the self-repulsion of the density.

    \int \int \rho_DM(r1) 1/|r1-r2| \rho_DM(r2) dr1 dr2

    Args:
        mol_or_mf (pyscf Mole or SCF): pyscf Mole or Mean Field object.
        dm (2D numpy ndarray): Density matrix.

    Returns:
        float: Self-repulsion energy (a.u).
    """
    if isinstance(mol_or_mf, gto.mole.Mole):
        j, _k = scf.hf.get_jk(mol_or_mf, dm)
    else:
        j, _k = mol_or_mf.get_jk()
    return np.einsum('ij,ij', j, dm)


def optimal_decomposition_error(self_repulsion, c, eri2c):
    r"""Compute the decomposition error for optimal density fitting.

    \int \int \rho_DM(r1) 1/|r1-r2| \rho_DF(r2) dr1 dr2

    Args:
        self_repulsion (float): Self-repulsion energy from the original density matrix.
        c (numpy ndarray): 1D array of density expansion coefficients.
        eri2c (numpy ndarray): 2D array of 2-center electron repulsion integrals.

    Returns:
        float: The decomposition error.

    Notes:
        - It is assumed that `c` are the optimal coefficients obtained from the density matrix.
        - `self_repulsion` can be set to 0 to avoid expensive computations when only the relative error is needed.
    """
    return self_repulsion - c @ eri2c @ c


def decomposition_error(self_repulsion, c, eri2c, eri3c, dm):
    r"""Compute the decomposition error for optimal density fitting.

    \int \int \rho_DM(r1) 1/|r1-r2| \rho_DF(r2) dr1 dr2

    Args:
        self_repulsion (float): Self-repulsion energy from the original density matrix.
        c (numpy ndarray): 1D array of density expansion coefficients.
        eri2c (numpy ndarray): 2D array of 2-center ERIs.
        eri3c (numpy ndarray): 3D array of 3-center ERIs.
        dm (numpy ndarray): Density matrix.

    Returns:
        float: The decomposition error.

    Notes:
        - If `c` are the optimal coefficients obtained from the density matrix, `optimal_decomposition_error()` can be used instead.
        - `self_repulsion` can be set to 0 to avoid expensive computations when only the relative error is needed.
    """
    projection = np.einsum('ijp,ij->p', eri3c, dm)
    return self_repulsion + c @ eri2c @ c - 2.0 * c @ projection


def _solve(A, v, slices=None):
    """Solve the linear system Ac = v, possibly in blocks.

    Args:
        A (numpy ndarray): Coefficient matrix.
        v (numpy ndarray): Right-hand side vector.
        slices (optional numpy ndarray): Assume that A is bloc-diagonal, by giving the boundaries of said blocks.

    Returns:
        numpy ndarray: Solution vector c.

    Raises:
        RuntimeError: If the `slices` argument is incorrectly formatted or inconsistent with the matrix
    """
    if slices is not None:
        if not (slices.ndim==2 and slices.shape[0]>0 and slices.shape[1]==2) or\
           not (slices[0,0] == 0 and slices[-1,1] == v.shape[0]):
            raise RuntimeError(f"Wrong argument {slices=}")

        c = np.empty_like(v)
        for s0,s1 in slices:
            c[s0:s1] = scipy.linalg.solve(A[s0:s1,s0:s1], v[s0:s1], assume_a='pos')
    else:
        c = scipy.linalg.solve(A, v, assume_a='pos')
    return c


def get_coeff(dm, eri2c, eri3c, slices=None):
    """Compute the density expansion coefficients.

    Args:
        dm (numpy ndarray): Density matrix.
        eri2c (numpy ndarray): 2-centers ERI matrix.
        eri3c (numpy ndarray): 3-centers ERI matrix.
        slices (optional numpy ndarray): Assume that eri2c is bloc-diagonal, by giving the boundaries of said blocks.

    Returns:
        A numpy ndarray containing the expansion coefficients of the density onto the auxiliary basis.
    """
    # Compute the projection of the density onto auxiliary basis using a Coulomb metric
    projection = np.einsum('ijp,ij->p', eri3c, dm)
    # Solve Jc = projection to get the coefficients
    return _solve(eri2c, projection, slices=slices)


def _get_inv_metric(mol, metric, v):
    """Compute the inverse metric applied to a vector.

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
    """Corrects decomposition coefficients to match the target number of electrons per atom.

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
    """Corrects decomposition coefficients to match the target total number of electrons.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        c0 (numpy ndarray): 1D array of initial decomposition coefficients.
        N (int, optional): Target number of electrons. If None, uses mol.nelectron. Defaults to None.
        mode (str): Correction method ('scale' or 'Lagrange'). Defaults to 'Lagrange'.
        metric (str): Metric type for Lagrange correction ('u', 's', or 'j'). Defaults to 'u'.

    Returns:
        numpy ndarray: Corrected decomposition coefficients (1D array).
    """
    mode = mode.lower()
    q = moments.r2_c(mol, None, moments=[0])[0]
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


def get_integrals_overlap(mol, auxmol, dm, grid_level=3):
    """Numerially compute overlap integrals.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object of the same molecule with an auxiliary basis set.
        dm (2D numpy array): Density matrix.
        grid_level (int): Level of the grid used to compute the numerical overlap integrals.

    Returns:
        Tuple containing:
        - float: self-overlap of the density,
        - 2D numpy array: overlap matrix (auxmol.nao,auxmol.nao) for the auxiliary basis,
        - 1D numpy array: projection of the density onto the auxiliary basis.
    """
    grid = make_grid_for_rho(mol, grid_level=grid_level)
    ao  = dft.numint.eval_ao(mol, grid.coords)
    rho = dft.numint.eval_rho(mol, ao, dm)
    auxao = dft.numint.eval_ao(auxmol, grid.coords).T
    auxao_w = np.einsum('px,x->px', auxao, grid.weights)
    proj    = np.einsum('px,x->p', auxao_w, rho)
    S       = np.einsum('px,qx->pq', auxao_w, auxao)
    self_S  = (rho*rho) @ grid.weights
    return self_S, S, proj


def decompose_overlap(mol, dm, auxbasis, slices=None, grid_level=3):
    """Fit molecular density onto an atom-centered basis using overlap metric.

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        dm (2D numpy array): Density matrix.
        auxbasis (string / pyscf basis dictionary): Atom-centered basis to decompose on.
        slices (optional numpy ndarray): Assume that eri2c is bloc-diagonal, by giving the boundaries of said blocks.
        grid_level (int): Level of the grid used to compute the numerical overlap integrals.

    Returns:
        Tuple containing:
        - copy of the pyscf Mole object with the auxbasis basis in a pyscf Mole object,
        - 1D numpy array containing the decomposition coefficients.
        - float: decomposition error in the overlap metric.
    """
    auxmol = compound.make_auxmol(mol, auxbasis)
    self_S, S, projection = get_integrals_overlap(mol, auxmol, dm, grid_level=grid_level)
    c = _solve(S, projection, slices=slices)
    err = self_S - c@projection
    return auxmol, c, err
