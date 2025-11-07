"""Excited state density and property analysis."""

import numpy as np
from . import moments


def get_cis(mf, nstates):
    """Wrapper for CIS (Configuration interaction singles) / TDA (Tamm-Dancoff approximation) computation.

    Args:
        mf: Pyscf mean-field object.
        nstates (int): Number of excited states to compute.

    Returns:
        TDA object: Converged TDA/CIS computation object with excited state information.
    """
    td = mf.TDA()
    td.nstates = nstates
    td.verbose = 5
    td.kernel()
    td.analyze()
    return td


def get_cis_tdm(td):
    """Extracts transition density matrices from TDA/CIS calculation.

    Args:
        td: TDA/CIS calculation object containing excitation amplitudes.

    Returns:
        numpy ndarray: Array of transition density matrices for all computed states.
    """
    return np.sqrt(2.0) * np.array([xy[0] for xy in td.xy])


def get_holepart(mol, x, coeff):
    """Computes the hole and particle density matrices (in AO basis) for a selected state.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x (numpy ndarray): Response vector (occ×virt) normalized to 1.
        coeff (numpy ndarray): Ground-state molecular orbital vectors.

    Returns:
        Two numpy ndarrays containing the hole density matrices and the particle density matrices respectively.
    """
    def mo2ao(mat, coeff):
        return np.dot(coeff, np.dot(mat, coeff.T))
    occ = mol.nelectron//2
    hole_mo = np.dot(x, x.T)
    part_mo = np.dot(x.T, x)
    hole_ao = mo2ao(hole_mo, coeff[:,:occ])
    part_ao = mo2ao(part_mo, coeff[:,occ:])
    return hole_ao, part_ao


def get_transition_dm(mol, x_mo, coeff):
    """Computes the transition density matrix for a selected state.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x_mo (numpy ndarray): Response vector (occ×virt) normalized to 1.
        coeff (numpy ndarray): Ground-state molecular orbital vectors.

    Returns:
        numpy ndarray: transition density matrix.
    """
    occ  = mol.nelectron//2
    x_ao = coeff[:,:occ] @ x_mo @ coeff[:,occ:].T
    return x_ao


def exciton_properties_c(mol, hole, part):
    """Computes the decomposed/predicted hole-particle distance, the hole size, and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole AO density decomposition coefficiants.
        part (numpy ndarray): Particle density decomposition coefficiants.

    Returns:
        Tuple of floats:
        - hole-particle distance,
        - hole size,
        - particle size.
    """
    _hole_N, hole_r, hole_r2 = moments.r2_c(mol, hole)
    _part_N, part_r, part_r2 = moments.r2_c(mol, part)

    dist = np.linalg.norm(hole_r-part_r)
    hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
    part_extent = np.sqrt(part_r2-part_r@part_r)
    return dist, hole_extent, part_extent


def exciton_properties_dm(mol, hole, part):
    """Computes the ab initio hole-particle distance, the hole size, and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        Tuple of floats:
        - hole-particle distance,
        - hole size,
        - particle size.
    """
    with mol.with_common_orig((0,0,0)):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)
    ao_r2 = mol.intor_symmetric('int1e_r2')

    hole_r = np.einsum('xij,ji->x', ao_r, hole)
    part_r = np.einsum('xij,ji->x', ao_r, part)
    hole_r2 = np.einsum('ij,ji', ao_r2, hole)
    part_r2 = np.einsum('ij,ji', ao_r2, part)

    dist = np.linalg.norm(hole_r-part_r)
    hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
    part_extent = np.sqrt(part_r2-part_r@part_r)
    return(dist, hole_extent, part_extent)


def exciton_properties(mol, hole, part):
    """Computes the ab initio or decomposed/predicted hole-particle distance, the hole size, and the particle size.

    Distance is defined as |<r>_hole - <r>_part|, and size as sqrt(<r^2> - <r>^2).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix in AO basis (2D) or decomposition coefficients (1D).
        part (numpy ndarray): Particle density matrix in AO basis (2D) or decomposition coefficients (1D).

    Returns:
        Tuple of floats:
        - hole-particle distance,
        - hole size,
        - particle size.

    Raises:
        RuntimeError: If the dimensions of hole and part do not match or are not 1D or 2D.
    """
    if hole.ndim==1 and part.ndim==1:
        return exciton_properties_c(mol, hole, part)
    elif hole.ndim==2 and part.ndim==2:
        return exciton_properties_dm(mol, hole, part)
    else:
        raise RuntimeError('Dimension mismatch')
