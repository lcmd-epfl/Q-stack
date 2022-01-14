import os
import numpy as np
from qstack import compound


def test_reader():
    
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)

    check_atom_object = np.array([[ 8, 20,  1, 23,  0,  0],[ 1, 24,  1, 27,  0,  0], [ 1, 28,  1, 31,  0,  0]], dtype=np.int32)

    assert mol.natm == 3
    assert mol.nelectron == 10
    assert type(mol.elements) == type([])
    assert np.array_equal(mol._atm, check_atom_object)

def test_makeauxmol():

    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    auxbasis = "cc-pvtz-jkfit"
    
    auxmol = compound.makeauxmol(mol, auxbasis)

    assert auxmol.natm == 3
    assert auxmol.nelectron == 10
    assert type(auxmol.elements) == type([])
    assert auxmol.basis == "cc-pvtz-jkfit"

def test_rotate_molecule():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    rotated_mol = compound.xyz_to_mol(path+'/data/rotated_H2O.xyz', 'def2svp', charge=0, spin=0)

    rotated = compound.rotate_molecule(mol, 90, 0, 0)
    assert np.linalg.norm(rotated_mol.atom_coords()-rotated.atom_coords()) < 1e-10

def test_mol_to_xyz():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)

    compound.mol_to_xyz(mol, path+'/data/H2O_saved.xyz')
