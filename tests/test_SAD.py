#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom


def test_water():
    print("Running water-test")
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'sto3g', charge=0, spin=0)

    Xsad = atom.get_repr(mol, ["H", "O"], 0, 0, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')

    print(np.array([*Xsad[:,1]]).shape)
    return 0


def test_other():
    print("Running weird-test")
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/645688c7c139c6414b5c0b00.xyz', 'sto3g', charge=-1, spin=2)

    Xsad = atom.get_repr(mol, ["H", "O", "C", "I", "Rh"], -1, 2, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='def2tzvpjkfit')

    print(np.array([*Xsad[:,1]]).shape)
    return 0


if __name__ == '__main__':
    test_water()
    test_other()
