#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.fields.decomposition import decompose, get_integrals, get_coeff


def test_fitting():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c0 = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    auxmol, c = decompose(mol, dm, 'cc-pvdz jkfit')
    assert(np.linalg.norm(c-c0)<1e-10)

def test_block_fitting():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')

    auxmol = compound.make_auxmol(mol, "cc-pvdz jkfit")
    S, eri2c, eri3c = get_integrals(mol, auxmol)
    atom_bounds = auxmol.aoslice_by_atom()[:,2:]
    eri2c0 = np.zeros_like(eri2c)
    for begin,end in atom_bounds:
        eri2c0[begin:end,begin:end] = eri2c[begin:end,begin:end]

    c0 = get_coeff(dm, eri2c0, eri3c)
    c = get_coeff(dm, eri2c0, eri3c, slices=atom_bounds)
    assert(np.linalg.norm(c-c0)<1e-10)


if __name__ == '__main__':
    test_fitting()
