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


def _orca2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform orca AO ordering into SA-GPR.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Re-arranged indices array.
    """
    #def _M1(n):
    #    return (n+1)//2 if n%2 else -((n+1)//2)
    idx = np.arange(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            for _n in range(nf):
                #for m in range(-l, l+1):
                #    m1 = _M1(m+l)
                #    idx[(i+(m1-m))] = i
                #    i+=1
                I = np.s_[i:i+msize]
                idx[I] = np.concatenate((idx[I][::-2], idx[I][1::2]))
                i += msize
    return idx


def _orca2gpr_sign(mol):
    """Given a molecule returns a list of multipliers needed to tranform from orca AO.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Array of +1/-1 multipliers.
    """
    signs = np.ones(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            if l<3:
                i += msize*nf
            else:
                for _n in range(nf):
                    signs[i+5:i+msize] = -1  # |m| >= 3
                    i+= msize
    return signs


def _pyscf2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Re-arranged indices array.
    """

    idx = np.arange(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            if l==1:
                for _n in range(nf):
                    idx[i:i+3] = [i+1,i+2,i]
                    i += 3
            else:
                i += msize * nf
    return idx


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

    def get_idx(mol, convention):
        convention = convention.lower()
        if convention == 'gpr':
            return np.arange(mol.nao)
        elif convention == 'pyscf':
            return _pyscf2gpr_idx(mol)
        elif convention == 'orca':
            return _orca2gpr_idx(mol)
        else:
            errstr = f'Conversion to/from the {convention} convention is not implemented'
            raise NotImplementedError(errstr)

    def get_sign(mol, convention):
        convention = convention.lower()
        if convention in ['gpr', 'pyscf']:
            return np.ones(mol.nao, dtype=int)
        elif convention == 'orca':
            return _orca2gpr_sign(mol)

    idx_src  = get_idx(mol, src)
    idx_dest = get_idx(mol, dest)
    sign_src  = get_sign(mol, src)
    sign_dest = get_sign(mol, dest)

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

