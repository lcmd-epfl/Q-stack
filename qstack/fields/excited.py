import numpy as np
import pyscf
from pyscf import scf, tdscf
from qstack import compound, fields

def get_cis(mf, nstates):
    td = mf.TDA()
    td.nstates = nstates
    td.verbose = 5
    td.kernel()
    td.analyze()
    return td

def get_cis_tdm(td):
    return np.sqrt(2.0) * np.array([ xy[0] for xy in td.xy ])

def get_holepart(mol, x, coeff):
    def mo2ao(mat, coeff):
        return np.dot(coeff, np.dot(mat, coeff.T))
    occ = mol.nelectron//2
    hole_mo = np.dot(x, x.T)
    part_mo = np.dot(x.T, x)
    hole_ao = mo2ao(hole_mo, coeff[:,:occ])
    part_ao = mo2ao(part_mo, coeff[:,occ:])
    return hole_ao, part_ao

def get_transition_dm(mol, x_mo, coeff):
    occ  = mol.nelectron//2
    x_ao = coeff[:,:occ] @ x_mo @ coeff[:,occ:].T
    return x_ao


def exciton_properties_c(mol, hole, part):
    hole_N, hole_r, hole_r2 = fields.moments.r2_c(hole, mol)
    part_N, part_r, part_r2 = fields.moments.r2_c(part, mol)

    dist = np.linalg.norm(hole_r-part_r)
    hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
    part_extent = np.sqrt(part_r2-part_r@part_r)
    return(dist, hole_extent, part_extent)

def exciton_properties_dm(mol, hole, part):
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
    if hole.ndim==1 and part.ndim==1:
        return exciton_properties_c(mol, hole, part)
    elif hole.ndim==2 and part.ndim==2:
        return exciton_properties_dm(mol, hole, part)
    else:
        raise RuntimeError('Dimension mismatch')
