#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.fields import decomposition, moments


def test_fitting():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c0 = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    _auxmol, c = decomposition.decompose(mol, dm, 'cc-pvdz jkfit')
    assert(np.linalg.norm(c-c0)<1e-10)


def test_block_fitting():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')

    auxmol = compound.make_auxmol(mol, "cc-pvdz jkfit")
    _, eri2c, eri3c = decomposition.get_integrals(mol, auxmol)
    atom_bounds = auxmol.aoslice_by_atom()[:,2:]
    eri2c0 = np.zeros_like(eri2c)
    for begin,end in atom_bounds:
        eri2c0[begin:end,begin:end] = eri2c[begin:end,begin:end]

    c0 = decomposition.get_coeff(dm, eri2c0, eri3c)
    c = decomposition.get_coeff(dm, eri2c0, eri3c, slices=atom_bounds)
    assert(np.linalg.norm(c-c0)<1e-10)


def test_fitting_error():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c0 = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    error0 = 4.876780263884939e-05
    auxmol = compound.make_auxmol(mol, 'cc-pvdz jkfit')
    eri2c = auxmol.intor('int2c2e_sph')
    self_repulsion = decomposition.get_self_repulsion(mol, dm)
    error = decomposition.decomposition_error(self_repulsion, c0, eri2c)
    assert(np.allclose(error, error0))


def test_fitting_noe():
    path = os.path.dirname(os.path.realpath(__file__))
    auxmol = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz jkfit', charge=0, spin=0)
    c = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    N = moments.r2_c(auxmol, c, moments=[0])[0]
    N0 = 10.000199558313856
    assert np.allclose(N,N0)


if __name__ == '__main__':
    test_fitting()
    test_block_fitting()
    test_fitting_error()
    test_fitting_noe()
