"""Symmetry operations for SPAHM(a,b) representations."""

import itertools
import numpy as np
from qstack import compound
from qstack.mathutils.matrix import sqrtm


def c_split_atom(mol, c, only_i=None):
    """Split coefficient vector by angular momentum quantum number for each atom.

    Organizes expansion coefficients into sublists grouped by angular momentum (l)
    for each atomic basis function.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        c (numpy ndarray): 1D array of expansion coefficients.
        only_i (list[int]): List of atom indices to use.

    Returns:
        list: List of coefficients (numpy ndarrays) per atom.
    """
    if only_i is None or len(only_i)==0:
        aoslice_by_atom = mol.aoslice_by_atom()[:,2:]
    else:
        aoslice_by_atom = mol.aoslice_by_atom()[only_i,2:]
    return [c[i0:i1] for i0, i1 in aoslice_by_atom]


def idxl0(i, l, m):
    """Return index of basis function (AO) with same L and N quantum numbers but M=0.

    Finds the m=0 component of the same angular momentum shell.

    Args:
        i (int): Basis function (atomic orbital) index.
        l (int): Angular momentum quantum number.
        m (int): Magnetic quantum number.

    Returns:
        int: Index of corresponding m=0 atomic orbital.
    """
    if l != 1:
        return i - m + l
    else:
        return i + [0, 2, 1][m]


def get_S(q, basis):
    """Compute overlap matrix and angular momentum info for an atom.

    Creates single-atom molecule and extracts basis function structure.

    Args:
        q (str): Element symbol.
        basis (str or dict): Basis set.

    Returns:
        tuple: (S, ao, ao_start) where:
        - S (numpy ndarray): Overlap matrix
        - ao (dict): Angular momentum info with 'l' and 'm' lists for each AO
        - ao_start (list): Starting indices for each angular momentum shell
    """
    mol = compound.make_atom(q, basis)
    S = mol.intor_symmetric('int1e_ovlp')
    (_, l, m), ao_start = compound.basis_flatten(mol, return_both=False, return_shells=True)
    ao = {'l': l, 'm': m}
    return S, ao, ao_start


def store_pair_indices_short(ao, ao_start):
    """Store basis function pair indices for m=0 components only.

    Creates list of (i,j) pairs using only the first basis function
    of each angular momentum shell, for compact representation.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO
        ao_start (list): Starting indices for each angular momentum shell.

    Returns:
        numpy ndarray: [i, j] index pairs for m=0 components with matching L.
    """
    l_shell = ao['l'][ao_start]
    idx = []
    for i, li in zip(ao_start, l_shell, strict=True):
        js = ao_start[np.where(l_shell==li)][:,None]
        idx.append(np.pad(js, ((0,0), (1,0)), mode='constant', constant_values=i))
    return np.vstack(idx)


def metric_matrix_short(idx, ao, S):
    """Compute metric matrix for symmetrization of short-format coefficients.

    Args:
        idx (numpy ndarray): [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix.
    """
    A = np.zeros((len(idx),len(idx)))
    for p, (i, j) in enumerate(idx):
        for p1, (i1, j1) in enumerate(idx[:p+1]):
            l  = ao['l'][i]
            l1 = ao['l'][i1]
            if l!=l1:
                continue
            A[p1,p] = A[p,p1] = S[i,i1] * S[j,j1] / (2*l+1)
    return sqrtm(A)


def vectorize_c(idx, c):
    """Vectorizes density fitting coefficients by forming products.

    Creates rotationally invariant representation from coefficient products.

    Args:
        idx (numpy ndarray): [i, j] basis function pair indices.
        c (numpy ndarray): 1D array of coefficients.

    Returns:
        numpy ndarray: 1D array of coefficient products c[i]*c[j].
    """
    v = np.zeros(len(idx))
    for p, (i, j) in enumerate(idx):
        v[p] = c[i]*c[j]
    return v


def vectorize_c_MR2021(idx_pair, ao, c):
    """Vectorizes coefficients using MR2021 scheme.

    Reference:
        J. T. Margraf, K. Reuter,
        "Pure non-local machine-learned density functional theory for electron correlation",
        Nat. Commun. 12, 344 (2021), doi:10.1038/s41467-020-20471-y

    Computes simplified rotationally invariant representation by contracting coefficients
    within each angular momentum shell.

    Args:
        idx_pair (numpy ndarray): [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        c (numpy ndarray): 1D array of density fitting coefficients.

    Returns:
        numpy ndarray: 1D array of contracted coefficient norms per shell.
    """
    idx = np.unique(idx_pair[:,0])
    v = np.zeros(len(idx))
    for p, i in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[i:i+msize]
    return v


def vectorize_c_short(idx, ao, c):
    """Vectorizes coefficients using short format with shell-wise dot products.

    Computes representation by contracting coefficient vectors of angular momentum shells.

    Args:
        idx (numpy ndarray): [i, j] basis function pair indices (shell starts).
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        c (numpy ndarray): 1D array of density fitting coefficients.

    Returns:
        numpy ndarray: 1D array of shell-pair dot products.
    """
    v = np.zeros(len(idx))
    for p, (i, j) in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[j:j+msize]
    return v


def store_pair_indices_z(ao):
    """Store basis function pairs with matching |m| quantum numbers.

    Creates list of all (i,j) pairs where basis functions have equal
    absolute values of magnetic quantum number m.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        numpy ndarray: [i, j] index pairs with |m_i| = |m_j|.
    """
    idx = []
    for i, mi in enumerate(ao['m']):
        js = np.where(abs(ao['m'])==abs(mi))[0][:,None]
        idx.append(np.pad(js, ((0,0), (1,0)), mode='constant', constant_values=i))
    return np.vstack(idx)


def store_pair_indices_z_only0(ao):
    """Store basis function pairs restricted to m=0 components only.

    Creates list of all (i,j) pairs where both basis functions have m=0.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        numpy ndarray: [i, j] index pairs where both m_i = m_j = 0.
    """
    i_m0 = np.where(ao['m']==0)[0]
    return np.array([*itertools.product(i_m0, i_m0)])


def metric_matrix_z(idx, ao, S):
    """Compute metric matrix for z-axis symmetric representations.

    Constructs metric matrix accounting for m and -m degeneracy. Matrix
    elements are nonzero only when angular momenta match and m quantum
    numbers satisfy m_i=m_i1 AND m_j=m_j1, or m_i=-m_i1 AND m_j=-m_j1.

    Args:
        idx (numpy ndarray): [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix for z-symmetric normalization.
    """
    A = np.zeros((len(idx),len(idx)))
    for p, (i, j) in enumerate(idx):
        for p1, (i1, j1) in enumerate(idx[:p+1]):
            li  = ao['l'][i ]
            lj  = ao['l'][j ]
            li1 = ao['l'][i1]
            lj1 = ao['l'][j1]
            if (li != li1) or (lj != lj1):
                continue
            mi  = ao['m'][i ]
            mj  = ao['m'][j ]
            mi1 = ao['m'][i1]
            mj1 = ao['m'][j1]

            A[p1,p] = A[p,p1] = ((mi==mi1)*(mj==mj1) + (mi==-mi1)*(mj==-mj1)*(mi!=0)) \
                                * S[idxl0(i,  li, mi), idxl0(i1, li1, mi1)] \
                                * S[idxl0(j , lj, mj), idxl0(j1, lj1, mj1)]

    return sqrtm(A)
