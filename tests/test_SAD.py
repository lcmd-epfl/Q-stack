#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom

def test_water_open_shell():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'sto3g', charge=1, spin=1) ## test breaks when effective open-shell caluclation is needed

    Xsad = atom.get_repr(mol, ["H", "O"], 1, 1, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    return Xsad

def test_water_close_shell():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'sto3g', charge=0, spin=0) ## test breaks when effective open-shell caluclation is needed

    Xsad = atom.get_repr(mol, ["H", "O"], 0, 0, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    return Xsad

def test_equivalence():
    Xos = test_water_open_shell()
    Xcs = test_water_close_shell()
    assert(not np.array_equal(Xos, Xcs))

if __name__ == '__main__':
    test_equivalence()
