#!/usr/bin/env python3

import os
import numpy as np
from pyscf import scf
from qstack import compound, fields
from qstack.io import turbomole


def _dipole_moment(mol, dm):
    return scf.hf.dip_moment(mol, dm, unit='au', origin=(0,0,0), verbose=0)


def test_turbomole_mos_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/turbomole/PUZWAI/PUZWAI.xyz', 'ccpvdz')
    mol_dip_true = np.array([-0.600499, 2.297241, -0.000000])  # from output file
    c, _e, _ = turbomole.read_mos(mol, path+'/data/turbomole/PUZWAI/mos')
    occ = np.zeros(mol.nao)
    occ[:mol.nelectron//2] = 2.0
    dm = fields.dm.make_rdm1(c, occ)
    mol_dip = _dipole_moment(mol, dm)
    assert np.allclose(mol_dip, mol_dip_true)


def test_turbomole_mos_reader_l4():
    path = os.path.dirname(os.path.realpath(__file__))
    mol_dip_true = {
            'ccpvdz': [-0.451381, 0.135083, 0.272028],
            'ccpvtz': [-0.439947, 0.130801, 0.264216],
            'ccpvqz': [-0.434773, 0.127902, 0.259653],
            }
    for basis, mol_dip0 in mol_dip_true.items():
        mol = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', basis)
        S = mol.intor('int1e_ovlp_sph')
        occ = np.zeros(mol.nao)
        occ[:mol.nelectron//2] = 2
        c, _e, _ = turbomole.read_mos(mol, f'{path}/data/turbomole/mos-{basis}')
        dm = fields.dm.make_rdm1(c, occ)
        mol_dip = _dipole_moment(mol, dm)
        assert abs(mol.nelectron-np.trace(dm @ S)) < 1e-8
        assert np.allclose(mol_dip, mol_dip0, atol=1e-6)


def test_turbomole_mos_reader_open():
    path = os.path.dirname(os.path.realpath(__file__))
    mol_dip0 = np.array([-0.250822, 0.052104, 0.126593])  # from output file, origin=(0,0,0)
    basis = 'ccpvdz'
    mol = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', basis, charge=1, spin=1)
    S = mol.intor('int1e_ovlp_sph')
    ca, _ea, _ = turbomole.read_mos(mol, f'{path}/data/turbomole/alpha')
    cb, _eb, _ = turbomole.read_mos(mol, f'{path}/data/turbomole/beta')
    occ = np.zeros((2, mol.nao))
    occ[0,:mol.nelec[0]] = 1
    occ[1,:mol.nelec[1]] = 1
    dma = fields.dm.make_rdm1(ca, occ[0])
    dmb = fields.dm.make_rdm1(cb, occ[1])
    dm = dma + dmb
    mol_dip = _dipole_moment(mol, dm)
    assert abs(mol.nelec[0]-np.trace(dma @ S)) < 1e-8
    assert abs(mol.nelec[1]-np.trace(dmb @ S)) < 1e-8
    assert np.allclose(mol_dip, mol_dip0, atol=1e-6)


if __name__ == '__main__':
    test_turbomole_mos_reader()
    test_turbomole_mos_reader_l4()
    test_turbomole_mos_reader_open()
