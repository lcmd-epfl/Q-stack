import numpy as np


def get_mrange(l):
    """Get the m quantum number range for a given angular momentum l.

    For l=1, returns pyscf order: x,y,z which is (1,-1,0).

    Args:
        l (int): Angular momentum quantum number.

    Returns:
        tuple or range: Magnetic quantum numbers for the given l.
    """
    if l==1:
        return (1,-1,0)
    else:
        return range(-l,l+1)


def _orca2gpr_idx(l, m):
    """Given a molecule returns a list of reordered indices to tranform orca AO ordering into SA-GPR.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Re-arranged indices array.
    """
    idx = np.arange(len(l))
    i=0
    while(i < len(idx)):
        msize = 2*l[i]+1
        j = np.s_[i:i+msize]
        idx[j] = np.concatenate((idx[j][::-2], idx[j][1::2]))
        i += msize
    signs = np.ones_like(idx)
    signs[np.where(np.abs(m)>=3)] = -1  # in pyscf order
    signs[idx] = signs # in orca order
    return idx, signs


def _pyscf2gpr_idx(l):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Re-arranged indices array.
    """

    idx = np.arange(len(l))
    i=0
    while(i < len(idx)):
        msize = 2*l[i]+1
        if l[i]==1:
            idx[i:i+3] = [i+1,i+2,i]
        i += msize
    return idx, np.ones_like(idx)


def reorder_ao(mol, vector, src='pyscf', dest='gpr'):
    """Reorder the atomic orbitals from one convention to another.

    For example, src=pyscf dest=gpr reorders p-orbitals from +1,-1,0 (pyscf convention)
    to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        vector (numpy.ndarray): Vector or matrix to reorder.
        src (str): Current convention. Defaults to 'pyscf'.
        dest (str): Convention to convert to (available: 'pyscf', 'gpr', 'orca'). Defaults to 'gpr'.

    Returns:
        numpy.ndarray: Reordered vector or matrix.

    Raises:
        NotImplementedError: If the specified convention is not implemented.
        ValueError: If vector dimension is not 1 or 2.
    """

    def get_idx(l, m, convention):
        convention = convention.lower()
        if convention == 'gpr':
            return np.arange(len(l)), np.ones_like(l)
        elif convention == 'pyscf':
            return _pyscf2gpr_idx(l)
        elif convention == 'orca':
            return _orca2gpr_idx(l, m)
        else:
            errstr = f'Conversion to/from the {convention} convention is not implemented'
            raise NotImplementedError(errstr)

    from .compound import basis_flatten
    _, l, m = basis_flatten(mol, return_both=False)
    idx_src, sign_src  = get_idx(l, m, src)
    idx_dest, sign_dest = get_idx(l, m, dest)

    if vector.ndim == 2:
        sign_src  = np.einsum('i,j->ij', sign_src, sign_src)
        sign_dest = np.einsum('i,j->ij', sign_dest, sign_dest)
        idx_dest = np.ix_(idx_dest,idx_dest)
        idx_src  = np.ix_(idx_src,idx_src)
    elif vector.ndim!=1:
        errstr = f'Dim = {vector.ndim} (should be 1 or 2)'
        raise ValueError(errstr)

    newvector = np.zeros_like(vector)
    newvector[idx_dest] = (sign_src*vector)[idx_src]
    newvector *= sign_dest

    return newvector

