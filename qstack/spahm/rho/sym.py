import numpy as np
from qstack import compound
from qstack.mathutils.matrix import sqrtm
from qstack.reorder import get_mrange


def idxl0(i, l, ao):
    """Returns index of basis function with same L and N quantum numbers but M=0.

    Finds the m=0 component of the same angular momentum shell for normalization.

    Args:
        i (int): Basis function index.
        l (int): Angular momentum quantum number.
        ao (dict): Angular momentum info dict with 'l' and 'm' keys.

    Returns:
        int: Index of corresponding m=0 basis function.
    """
    # return the index of the basis function with the same L and N but M=0
    if l != 1:
        return i - ao['m'][i]+l
    else:
        return i + [0, 2, 1][ao['m'][i]]

def get_S(q, basis):
    """Computes overlap matrix and angular momentum info for an atom.

    Creates single-atom molecule and extracts basis function structure.

    Args:
        q (str): Element symbol.
        basis (str or dict): Basis set specification.

    Returns:
        tuple: (S, ao, ao_start) where:
            - S (numpy ndarray): Overlap matrix
            - ao (dict): Angular momentum info with 'l' and 'm' lists
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
        ao (dict): Angular momentum info with 'l' and 'm' keys.

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
        idx (list): List of basis function pair indices.
        ao (dict): Angular momentum info.
        S (numpy ndarray): Overlap matrix.

    Returns:
        numpy ndarray: Square root of metric matrix for normalization.
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
    idx = sorted(set(np.array(idx_pair)[:,0]))
    v = np.zeros(len(idx))
    for p,i in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[i:i+msize]
    return v


def vectorize_c_short(idx, ao, c):
    v = np.zeros(len(idx))
    for p, [i,j] in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[j:j+msize]
    return v


def store_pair_indices_z(ao):
    idx = []
    for i, mi in enumerate(ao['m']):
        for j, mj in enumerate(ao['m']):
            if abs(mi)!=abs(mj):
                continue
            idx.append([i,j])
    return idx


def store_pair_indices_z_only0(ao):
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
