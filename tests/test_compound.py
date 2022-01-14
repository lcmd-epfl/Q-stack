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