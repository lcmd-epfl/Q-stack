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

def test_water_SAD_guess_open_shell():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'sto3g', charge=1, spin=1) ## test breaks when effective open-shell caluclation is needed

    Xsad = atom.get_repr(mol, ["H", "O"], 1, 1, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    Xtrue = np.load(path+'/data/SPAHM_a_H2O/X_H2O-RC_SAD.npy', allow_pickle=True)
    assert(Xsad.shape == Xtrue.shape)
    for a, a_true in zip(Xsad, Xtrue):
        assert(a[0] == a_true[0])                        # atom type
        assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations

def test_water_SAD_guess_close_shell():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'sto3g', charge=0, spin=0) ## test breaks when effective open-shell caluclation is needed

    Xsad = atom.get_repr(mol, ["H", "O"], 0, None, dm=None,
                      xc = 'hf', guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    Xtrue = np.load(path+'/data/SPAHM_a_H2O/X_H2O_SAD.npy', allow_pickle=True)
    assert(Xsad.shape == Xtrue.shape)
    for a, a_true in zip(Xsad, Xtrue):
        assert(a[0] == a_true[0])                        # atom type
        assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations

def test_water_single_element():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'minao', charge=0, spin=None)

    X = atom.get_repr(mol, ["H", "O"], 0, None, dm=None,
                      guess='LB', model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=['O']) #requesting reps for O-atom only

    X_true = np.load(path+'/data/SPAHM_a_H2O/X_H2O.npy', allow_pickle=True)
    a = X[0]
    assert(X.shape == np.array(X_true[0], ndmin=2).shape)
    for a_true in X_true:
        if a[0] == a_true[0]:                       # atom type
            assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations



if __name__ == '__main__':
    test_water()
    test_water_SAD_guess_close_shell()
    test_water_SAD_guess_open_shell()
    test_water_single_element()
