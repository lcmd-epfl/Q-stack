#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound


def test_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0, unit='A')
    check_atom_coord = np.array([[ 0. ,         0.        ,  0.22259084], [ 0. ,         1.4275936 , -0.89036336], [ 0. ,        -1.4275936 , -0.89036336]])
    assert mol.natm == 3
    assert mol.nelectron == 10
    assert mol.elements == ['O', 'H', 'H']
    assert np.linalg.norm(mol.atom_coords()-check_atom_coord) < 1e-8


def test_reader_ignore():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/dori/H6CN____monA_0012.xyz', ignore=True)
    assert mol.charge == -1
    assert mol.spin == 0


def test_makeauxmol():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    auxbasis = "cc-pvtz-jkfit"
    auxmol = compound.make_auxmol(mol, auxbasis)
    assert auxmol.natm == 3
    assert auxmol.nelectron == 10
    assert type(auxmol.elements) is type([])
    assert auxmol.basis == "cc-pvtz-jkfit"


def test_rotate_molecule():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    rotated_mol = compound.xyz_to_mol(path+'/data/rotated_H2O.xyz', 'def2svp', charge=0, spin=0)
    rotated = compound.rotate_molecule(mol, 90, 0, 0)
    assert np.linalg.norm(rotated_mol.atom_coords()-rotated.atom_coords()) < 1e-10


def test_mol_to_xyz():
    path = os.path.dirname(os.path.realpath(__file__))
    molpath = path+'/data/H2O_saved.xyz'
    with open(molpath) as f:
        xyz0 = f.read().strip()
    mol = compound.xyz_to_mol(molpath, 'def2svp', charge=0, spin=0)
    xyz = compound.mol_to_xyz(mol, '/dev/null')
    assert np.all(xyz == xyz0)


def test_commentline():
    path = os.path.dirname(os.path.realpath(__file__))
    names = ["HO_json.xyz", "HO_keyvalcomma.xyz", "HO_keyvalspace.xyz", "HO_spinline.xyz"]
    for name in names:
        mol = compound.xyz_to_mol(os.path.join(path,'data',name), 'def2svp', parse_comment=True)
        assert mol.spin == 0
        assert mol.charge == -1


if __name__ == '__main__':
    test_reader()
    test_reader_ignore()
    test_makeauxmol()
    test_rotate_molecule()
    test_mol_to_xyz()
    test_commentline()
