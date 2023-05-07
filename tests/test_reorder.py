#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound, tools
from qstack.fields.decomposition import decompose
from qstack.math.matrix import from_tril


def test_reorder_pyscf_gpr():
    path = os.path.dirname(os.path.realpath(__file__))

    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    dm1 = tools.reorder_ao(mol, dm,  src='pyscf', dest='gpr')
    dm2 = tools.reorder_ao(mol, dm1, src='gpr', dest='pyscf')
    assert(np.linalg.norm(dm-dm2)==0)

    auxmol = compound.make_auxmol(mol, 'cc-pvdz jkfit')
    c  = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    c1 = tools.reorder_ao(auxmol, c,  src='pyscf', dest='gpr')
    c2 = tools.reorder_ao(auxmol, c1, src='gpr', dest='pyscf')
    assert(np.linalg.norm(c-c2)==0)


def test_reorder_pyscf_gpr_orca():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/reorder/2_3FOD.xyz', 'ccpvqz', charge=0, spin=0)
    dm_orca  = from_tril(np.fromfile(path+'/data/reorder/2_3FOD.scfp'))
    dm_gpr   = from_tril(np.load(path+'/data/reorder/2_3FOD.gpr.dm.npy'))
    dm_pyscf = from_tril(np.load(path+'/data/reorder/2_3FOD.pyscf.dm.npy'))

    dm_gpr1   = tools.reorder_ao(mol, dm_orca, 'orca', 'gpr')
    assert(np.linalg.norm(dm_gpr1-dm_gpr)==0)
    dm_gpr1   = tools.reorder_ao(mol, dm_pyscf, 'pyscf', 'gpr')
    assert(np.linalg.norm(dm_gpr1-dm_gpr)==0)

    dm_pyscf1 = tools.reorder_ao(mol, dm_orca, 'orca', 'pyscf')
    assert(np.linalg.norm(dm_pyscf1-dm_pyscf)==0)
    dm_pyscf1 = tools.reorder_ao(mol, dm_gpr, 'gpr', 'pyscf')
    assert(np.linalg.norm(dm_pyscf1-dm_pyscf)==0)

    dm_orca1 = tools.reorder_ao(mol, dm_pyscf, 'pyscf', 'orca')
    assert(np.linalg.norm(dm_orca1-dm_orca)==0)
    dm_orca1 = tools.reorder_ao(mol, dm_gpr, 'gpr', 'orca')
    assert(np.linalg.norm(dm_orca1-dm_orca)==0)


if __name__ == '__main__':
    test_reorder_pyscf_gpr()
    test_reorder_pyscf_gpr_orca()
