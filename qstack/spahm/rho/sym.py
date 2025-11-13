"""Symmetry operations for SPAHM(a,b) representations."""

import numpy as np
from qstack import compound
from qstack.mathutils.matrix import sqrtm
from qstack.reorder import get_mrange


def idxl0(ao_i, l, ao):
    """Returns index of basis function with same L and N quantum numbers but M=0.

    Finds the m=0 component of the same angular momentum shell.

    Args:
        ao_i (int): Basis function (atomic orbital) index.
        l (int): Angular momentum quantum number.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        int: Index of corresponding m=0 basis function.
    """
    if l != 1:
        return ao_i - ao['m'][ao_i]+l
    else:
        return ao_i + [0, 2, 1][ao['m'][ao_i]]


def get_S(q, basis):
    """Computes overlap matrix and angular momentum info for an atom.

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

    l_per_bas, _n_per_bas, ao_start = compound.singleatom_basis_enumerator(mol._basis[q])

    ao = {'l': [], 'm': []}
    for l in l_per_bas:
        ao['l'].extend([l]*(2*l+1))
        ao['m'].extend(get_mrange(l))

    return S, ao, ao_start


def store_pair_indices(ao):
    """Stores basis function pair indices with matching L and M quantum numbers.

    Creates list of all (i,j) pairs where basis functions have identical angular momenta.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        list: List of [i, j] index pairs with matching (l, m).
    """
    idx = []
    for i, [li, mi] in enumerate(zip(ao['l'], ao['m'], strict=True)):
        for j, [lj, mj] in enumerate(zip(ao['l'], ao['m'], strict=True)):
            if (li!=lj) or (mi!=mj):
                continue
            idx.append([i, j])
    return idx


def store_pair_indices_short(ao, ao_start):
    """Stores basis function pair indices for m=0 components only.

    Creates list of (i,j) pairs using only the first basis function (m=0)
    of each angular momentum shell, for compact representation.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO
        ao_start (list): Starting indices for each angular momentum shell.

    Returns:
        list: List of [i, j] index pairs for m=0 components with matching L.
    """
    idx = []
    for i in ao_start:
        for j in ao_start:
            li = ao['l'][i]
            lj = ao['l'][j]
            if li!=lj:
                continue
            idx.append([i, j])
    return idx


def metric_matrix(q, idx, ao, S):
    """Computes metric matrix for symmetrization of density fitting coefficients.

    Constructs metric matrix from overlap integrals of basis function pairs,
    normalized by angular momentum degeneracy (2l+1). Returns square root
    for transformation to orthonormal representation.

    Args:
        q (str): Element symbol key for angular momentum info.
        idx (list): List of [i, j] basis function pair indices.
        ao (dict): Angular momentum info dict with nested structure ao[q].
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix.
    """
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i,  j  = idx[p]
            i1, j1 = idx[p1]
            l  = ao['l'][i]
            l1 = ao['l'][i1]
            if(l!=l1):
                continue
            A[p1,p] = A[p,p1] = 1.0/(2*l+1) \
                                * S[idxl0(i, l, ao[q]), idxl0(i1, l, ao[q])] \
                                * S[idxl0(j, l, ao[q]), idxl0(j1, l, ao[q])]
    return sqrtm(A)


def metric_matrix_short(idx, ao, S):
    """Computes metric matrix for symmetrization of short-format coefficients.

    Args:
        idx (list): List of [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix.
    """
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i, j  = idx[p]
            i1,j1 = idx[p1]
            l  = ao['l'][i]
            l1 = ao['l'][i1]
            if(l!=l1):
                continue
            A[p1,p] = A[p,p1] = S[i,i1] * S[j,j1] / (2*l+1)
    return sqrtm(A)


def vectorize_c(idx, c):
    """Vectorizes density fitting coefficients by forming products.

    Creates rotationally invariant representation from coefficient products.

    Args:
        idx (list): List of [i, j] basis function pair indices.
        c (numpy ndarray): 1D array of coefficients.

    Returns:
        numpy ndarray: 1D array of coefficient products c[i]*c[j].
    """
    v = np.zeros(len(idx))
    for p, (i,j) in enumerate(idx):
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
        idx_pair (list): List of [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        c (numpy ndarray): 1D array of density fitting coefficients.

    Returns:
        numpy ndarray: 1D array of contracted coefficient norms per shell.
    """
    idx = sorted(set(np.array(idx_pair)[:,0]))
    v = np.zeros(len(idx))
    for p,i in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[i:i+msize]
    return v


def vectorize_c_short(idx, ao, c):
    """Vectorizes coefficients using short format with shell-wise dot products.

    Computes representation by contracting coefficient vectors of angular momentum shells.

    Args:
        idx (list): List of [i, j] basis function pair indices (shell starts).
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        c (numpy ndarray): 1D array of density fitting coefficients.

    Returns:
        numpy ndarray: 1D array of shell-pair dot products.
    """
    v = np.zeros(len(idx))
    for p, [i,j] in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[j:j+msize]
    return v


def store_pair_indices_z(ao):
    """Stores basis function pairs with matching |m| quantum numbers.

    Creates list of all (i,j) pairs where basis functions have equal
    absolute values of magnetic quantum number m.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        list: List of [i, j] index pairs with |m_i| = |m_j|.
    """
    idx = []
    for i, mi in enumerate(ao['m']):
        for j, mj in enumerate(ao['m']):
            if abs(mi)!=abs(mj):
                continue
            idx.append([i,j])
    return idx


def store_pair_indices_z_only0(ao):
    """Stores basis function pairs restricted to m=0 components only.

    Creates list of all (i,j) pairs where both basis functions have m=0.

    Args:
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.

    Returns:
        list: List of [i, j] index pairs where both m_i = m_j = 0."""
    idx = []
    for i, mi in enumerate(ao['m']):
        if mi!=0:
            continue
        for j, mj in enumerate(ao['m']):
            if mj!=0:
                continue
            idx.append([i,j])
    return idx


def metric_matrix_z(idx, ao, S):
    """Computes metric matrix for z-axis symmetric representations.

    Constructs metric matrix accounting for m and -m degeneracy. Matrix
    elements are nonzero only when angular momenta match and m quantum
    numbers satisfy m_i=m_j AND m_i1=m_j1, or m_i=-m_j AND m_i1=-m_j1.

    Args:
        idx (list): List of [i, j] basis function pair indices.
        ao (dict): Angular momentum info with 'l' and 'm' lists for each AO.
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix for z-symmetric normalization."""
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i,i1 = idx[p]
            j,j1 = idx[p1]
            li  = ao['l'][i ]
            li1 = ao['l'][i1]
            lj  = ao['l'][j ]
            lj1 = ao['l'][j1]
            if (li != lj) or (li1 != lj1):
                continue

            mi  = ao['m'][i ]
            mi1 = ao['m'][i1]
            mj  = ao['m'][j ]
            mj1 = ao['m'][j1]

            A[p1,p] = A[p,p1] = ((mi==mj)*(mi1==mj1) + (mi==-mj)*(mi1==-mj1)*(mi!=0)) \
                                * S[idxl0(i,  li,  ao), idxl0(j,  li,  ao)] \
                                * S[idxl0(i1, li1, ao), idxl0(j1, li1, ao)]

    return sqrtm(A)
