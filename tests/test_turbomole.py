#!/usr/bin/env python3

import os
import numpy as np
from pyscf import scf
from pyscf.data import elements
from qstack import compound, fields
from qstack.io import turbomole


def _dipole_moment(mol, dm):
    coords = mol.atom_coords()
    mass = np.array(elements.MASSES)[compound.numbers(mol)]
    mass_center = np.einsum('i,ix->x', mass, coords) / sum(mass)
    return scf.hf.dip_moment(mol, dm, unit='au', origin=mass_center, verbose=0)


def test_turbomole_mos_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/PUZWAI/PUZWAI.xyz', 'ccpvdz')
    mol_dip_true = np.array([-0.600499, 2.297241, -0.000000])  # from output file
    c, _e = turbomole.read_mos(mol, path+'/data/PUZWAI/mos')
    occ = np.zeros(mol.nao)
    occ[:mol.nelectron//2] = 2.0
    dm = fields.dm.make_rdm1(c, occ)
    mol_dip = _dipole_moment(mol, dm)
    assert np.allclose(mol_dip, mol_dip_true)


if __name__ == '__main__':
    test_turbomole_mos_reader()
