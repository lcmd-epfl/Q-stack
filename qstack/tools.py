import numpy as np

def _pyscf2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR .

    Args:
        mol (pyscf Mole): a pyscf Mole object.
        

    Returns:
        numpy ndarray: Array of re-arranged indices.
    """

    idx = np.arange(mol.nao_nr(), dtype=int)

    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]

        numbs = np.zeros(max_l+1, dtype=int)
        for gto in mol._basis[q]:
            l = gto[0]
            nf = max([len(prim)-1 for prim in gto[1:]])
            numbs[l] += nf

        i+=numbs[0]
        if(max_l<1):
            continue  

        for n in range(numbs[1]):
            idx[i  ] = i+1
            idx[i+1] = i+2
            idx[i+2] = i
            i += 3

        for l in range(2, max_l+1):
            i += (2*l+1)*numbs[l]

    return idx


def _gpr2pyscf_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform SA-GPR AO ordering into pyscf.

    Args:
        mol (pyscf Mole): a pyscf Mole object.
        

    Returns:
        numpy ndarray: Array of re-arranged indices.
    """

    idx = np.arange(mol.nao_nr(), dtype=int)

    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]

        numbs = np.zeros(max_l+1, dtype=int)
        for gto in mol._basis[q]:
            l = gto[0]
            nf = max([len(prim)-1 for prim in gto[1:]])
            numbs[l] += nf

        i+=numbs[0]
        if(max_l<1):
            continue

        for n in range(numbs[1]):
            idx[i+1] = i
            idx[i+2] = i+1
            idx[i  ] = i+2
            i += 3

        for l in range(2, max_l+1):
            i += (2*l+1)*numbs[l]

    return idx


def pyscf2gpr(mol, vector):
    """Reorder p-orbitals from +1,-1,0 (pyscf convention) to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf Mole): a pyscf Mole object.
        vector (numpy ndarray): a vector or a matrix of atomic orbitals ordered in pyscf convention.

    Returns:
        numpy ndarray: The reordered vector or matrix.
    """

    idx = _pyscf2gpr_idx(mol)
    dim = vector.ndim

    if dim == 1:
        return vector[idx]
    elif dim == 2:
        return vector[idx].T[idx]
    else:
        errstr = 'Dim = '+ str(dim)+' (should be 1 or 2)'
        raise Exception(errstr)

def gpr2pyscf(mol, vector):
    """Reorder p-orbitals from -1,0,+1 (SA-GPR convention) to +1,-1,0 (pyscf convention).

    Args:
        mol (pyscf Mole): a pyscf Mole object.
        vector (numpy ndarray): a vector or a matrix of atomic orbitals ordered in pyscf convention.

    Returns:
        numpy ndarray: The reordered vector or matrix.
    """

    idx = _gpr2pyscf_idx(mol)
    dim = vector.ndim

    if dim == 1:
        return vector[idx]
    elif dim == 2:
        return vector[idx].T[idx]
    else:
        errstr = 'Dim = '+ str(dim)+' (should be 1 or 2)'
        raise Exception(errstr)
