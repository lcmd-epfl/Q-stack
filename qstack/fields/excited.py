import numpy as np
import pyscf
from pyscf import scf, tdscf
from qstack import compound
from qstack.fields import moments

def get_cis(mf, nstates):
    """

    .. todo::
        Write the complete docstring.
    """
    td = mf.TDA()
    td.nstates = nstates
    td.verbose = 5
    td.kernel()
    td.analyze()
    return td

def get_cis_tdm(td):
    """

    .. todo::
        Write the complete docstring.
    """
    return np.sqrt(2.0) * np.array([ xy[0] for xy in td.xy ])

def get_holepart(mol, x, coeff):
    """Computes the hole and particle density matrices (atomic orbital basis) of selected states.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x (numpy ndarray): Response vector (nstates×occ×virt) normalized to 1.
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
    """ Compute the Transition Density Matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        x_mo (numpy ndarray): Response vector (nstates×occ×virt) normalized to 1.
        coeff (numpy ndarray): Ground-state molecular orbital vectors.

    Returns:
        A numpy ndarray containing the Transition Density Matrix.
    """

    occ  = mol.nelectron//2
    x_ao = coeff[:,:occ] @ x_mo @ coeff[:,occ:].T
    return x_ao


def exciton_properties_c(mol, hole, part):
    """ Computes the decomposed/predicted hole-particle distance, the hole size and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        Three floats: the hole-particle distance, the hole size, and the particle size respectively.
    """

    hole_N, hole_r, hole_r2 = moments.r2_c(hole, mol)
    part_N, part_r, part_r2 = moments.r2_c(part, mol)

    dist = np.linalg.norm(hole_r-part_r)
    hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
    part_extent = np.sqrt(part_r2-part_r@part_r)
    return(dist, hole_extent, part_extent)

def exciton_properties_dm(mol, hole, part):
    """Computes the ab initio hole-particle distance, the hole size and the particle size.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        Three floats: the hole-particle distance, the hole size, and the particle size respectively.
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
    """Computes the ab initio or decomposed/predicted hole-particle distance, the hole size and the particle size according to the number of dimensions of the density matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        hole (numpy ndarray): Hole density matrix.
        part (numpy ndarray): Particle density matrix.

    Returns:
        The hole-particle distance, the hole size, and the particle size as floats. 
    """

    if hole.ndim==1 and part.ndim==1:
        return exciton_properties_c(mol, hole, part)
    elif hole.ndim==2 and part.ndim==2:
        return exciton_properties_dm(mol, hole, part)
    else:
        raise RuntimeError('Dimension mismatch')
