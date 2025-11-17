"""Functions for reordering atomic orbitals between different conventions.

Provides:
    pyscf2gpr_l1_order: indices to reorder l=1 orbitals from PySCF to GPR.
"""

import numpy as np
from .tools import slice_generator


pyscf2gpr_l1_order = [1,2,0]


def get_mrange(l):
    """Get the m quantum number range for a given angular momentum l.

    For l=1, returns pyscf order: x,y,z which is (1,-1,0).
    For other l, returns the standard range from -l to +l.

    Args:
        l (int): Angular momentum quantum number.

    Returns:
        tuple or range: Magnetic quantum numbers for the given l.
    """
    if l==1:
        return (1,-1,0)
    else:
        return range(-l,l+1)


def _orca2gpr_idx(l_slices, m):
    """Given a molecule returns a list of reordered indices to tranform Orca AO ordering into SA-GPR.

    In Orca, orbital ordering corresponds to:
        m=0, +1, -1, +2, -2, ..., +l, -l
    while in SA-GPR it is:
        m=-l, -l+1, ..., -1, 0, +1, ..., l-1, l
    Additionally, Orca uses a different sign convention for |m|>=3.

    Args:
        l_slices (iterator): Iterator that yeilds (l: int, s: slice) per shell, where
            l is angular momentum quantum number and s is the corresponding slice of size 2*l+1.
        m (np.ndarray): Array of magnetic quantum numbers per AO.

    Returns:
        tuple: Re-arranged indices array and sign array.
    """
    idx = np.arange(len(m))
    for _l, s in l_slices:
        idx[s] = np.concatenate((idx[s][::-2], idx[s][1::2]))
    signs = np.ones_like(idx)
    signs[np.where(np.abs(m)>=3)] = -1  # in pyscf order
    signs[idx] = np.copy(signs)  # in orca order. copy for numpy < 2
    return idx, signs


def _pyscf2gpr_idx(l_slices, m):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    In SA-GPR, orbital ordering corresponds to:
        m=-l, -l+1, ..., -1, 0, +1, ..., l-1, l
    In PySCF, it is the same except for p-orbitals which are ordered as:
        m=+1, -1, 0 (i.e., x,y,z).
    Signs are the same in both conventions, so they are returned for compatibility.

    Args:
        l_slices (iterator): Iterator that yeilds (l: int, s: slice) per shell, where
            l is angular momentum quantum number and s is the corresponding slice of size 2*l+1.
        m (np.ndarray): Array of magnetic quantum numbers per AO.

    Returns:
        tuple: Re-arranged indices array and sign array.
    """
    idx = np.arange(len(m))
    for l, s in l_slices:
        if l==1:
            idx[s] = idx[s][pyscf2gpr_l1_order]
    return idx, np.ones_like(idx)


def _gau2gpr_idx(l_slices, m):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    In SA-GPR, orbital ordering corresponds to:
        m=-l, -l+1, ..., -1, 0, +1, ..., l-1, l
    In Gaussian, it is:
        l=1: m=+1, -1, 0 (i.e., x,y,z, like in PySCF)
        l>1: m=0, +1, -1, +2, -2, ..., +l, -l (like in Orca)
    Signs are the same in both conventions, so they are returned for compatibility.

    Args:
        l_slices (iterator): Iterator that yeilds (l: int, s: slice) per shell, where
            l is angular momentum quantum number and s is the corresponding slice of size 2*l+1.
        m (np.ndarray): Array of magnetic quantum numbers per AO.

    Returns:
        tuple: Re-arranged indices array and sign array.
    """
    idx = np.arange(len(m))
    for l, s in l_slices:
        if l==1:
            idx[s] = idx[s][pyscf2gpr_l1_order]
        elif l>1:
            idx[s] = np.concatenate((idx[s][::-2], idx[s][1::2]))
    return idx, np.ones_like(idx)


def reorder_ao(mol, vector, src='pyscf', dest='gpr'):
    """Reorder the atomic orbitals from one convention to another.

    For example, src=pyscf dest=gpr reorders p-orbitals from +1,-1,0 (pyscf convention)
    to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        vector (numpy.ndarray): Vector (nao,) or matrix (mol.nao,mol.nao) to reorder.
            If None, returns the indices to reorder and sign multipliers for an 1D vector
            to use as `x = x[idx]*sign`.
        src (str): Current convention. Defaults to 'pyscf'.
        dest (str): Convention to convert to (available: 'pyscf', 'gpr', 'orca', 'gaussian'). Defaults to 'gpr'.

    Returns:
        numpy.ndarray: Reordered vector or matrix, or tuple of (idx (numpy.ndarray), sign (numpy.ndarray)) if vector is None.

    Raises:
        NotImplementedError: If the specified convention is not implemented.
        ValueError: If vector dimension is not 1 or 2.
    """
    def get_idx(l_shells, m, convention):
        convention = convention.lower()
        l_slices = slice_generator(l_shells, inc=lambda l: 2*l+1)
        if convention == 'gpr':
            return np.arange(len(m)), np.ones_like(m)
        elif convention == 'pyscf':
            return _pyscf2gpr_idx(l_slices, m)
        elif convention == 'orca':
            return _orca2gpr_idx(l_slices, m)
        elif convention == 'gaussian':
            return _gau2gpr_idx(l_slices, m)
        else:
            errstr = f'Conversion to/from the {convention} convention is not implemented'
            raise NotImplementedError(errstr)

    if vector is not None and vector.ndim not in (1,2):
        errstr = f'Dim = {vector.ndim} (should be 1 or 2)'
        raise ValueError(errstr)

    if src.lower() == dest.lower():
        if vector is None:
            return np.arange(mol.nao), np.ones(mol.nao)
        else:
            return vector

    from .compound import basis_flatten

    (_, l, m), shell_start = basis_flatten(mol, return_both=False, return_shells=True)
    l_shells = l[shell_start]

    idx_src, sign_src  = get_idx(l_shells, m, src)
    idx_dest, sign_dest = get_idx(l_shells, m, dest)

    if vector is None:
        idx = np.arange(mol.nao)
        idx[idx_dest] = idx[idx_src]
        sign = np.ones_like(idx)
        sign[idx_dest] = (sign_src*sign)[idx_src] * sign_dest[idx_dest]
        return idx, sign

    if vector.ndim == 2:
        sign_src  = np.einsum('i,j->ij', sign_src, sign_src)
        sign_dest = np.einsum('i,j->ij', sign_dest, sign_dest)
        idx_dest = np.ix_(idx_dest,idx_dest)
        idx_src  = np.ix_(idx_src,idx_src)

    newvector = np.zeros_like(vector)
    newvector[idx_dest] = (sign_src*vector)[idx_src] * sign_dest[idx_dest]
    return newvector
