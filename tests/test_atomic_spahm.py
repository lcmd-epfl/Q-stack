#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom


def test_water():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'minao', charge=0, spin=None)

    X = atom.get_repr(mol, ["H", "O"], 0, None, dm=None,
                      guess='LB', model='lowdin-long-x', auxbasis='ccpvdzjkfit')

    X_true = np.load(path+'/data/SPAHM_a_H2O/X_H2O.npy', allow_pickle=True)
    assert(X.shape == X_true.shape)
    for a, a_true in zip(X, X_true):
        assert(a[0] == a_true[0])                        # atom type
        assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations


if __name__ == '__main__':
    test_water()
