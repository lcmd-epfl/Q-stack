#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom

PATH = os.path.dirname(os.path.realpath(__file__))

def underlying_test(true_data_relpath, X):
    X_true = np.load(PATH+true_data_relpath, allow_pickle=True)
    assert(X.shape == X_true.shape)
    for a, a_true in zip(X, X_true, strict=True):
        assert(a[0] == a_true[0])                        # atom type
        assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations

def test_water():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O.npy', X)

def test_water_alternate():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    #X = atom.get_repr(mol, ["H", "O"], None, dm=None,
    #                  guess='LB', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    X = atom.get_repr("atom", [mol], [PATH], 'LB', spin=[None], auxbasis='ccpvdzjkfit', with_symbols=True)
    underlying_test('/data/SPAHM_a_H2O/X_H2O.npy', X)

def test_water_lowdinshortx():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-short-x', auxbasis='ccpvdzjkfit')
    X = np.array([(z,np.trim_zeros(v)) for z,v in X], dtype=object) ## trimming is necessary to get the short-version vector !
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-short-x.npy', X)

def test_water_lowdinlong():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-long', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-long.npy', X)

def test_water_lowdinshort():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-short', auxbasis='ccpvdzjkfit')
    X = np.array([(z,np.trim_zeros(v)) for z,v in X], dtype=object) ## trimming is necessary to get the short-version vector !
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-short.npy', X)

def test_water_mr21():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='MR2021', auxbasis='ccpvdzjkfit')
    X = np.array([(z,np.trim_zeros(v)) for z,v in X], dtype=object) ## trimming is necessary to get the short-version vector !
    underlying_test('/data/SPAHM_a_H2O/X_H2O_MR2021.npy', X)

def test_water_SAD_guess_open_shell():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=1, spin=1) ## test breaks when effective open-shell caluclation is needed
    Xsad = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=[1], with_symbols=True,
                         xc = 'hf', model='sad-diff', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O-RC_SAD.npy', Xsad)

def test_water_SAD_guess_close_shell():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=0, spin=0) ## test breaks when effective open-shell caluclation is needed
    Xsad = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=None, with_symbols=True,
                         xc = 'hf', model='sad-diff', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O_SAD.npy', Xsad)

def test_water_single_element():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=['O']) #requesting reps for O-atom only
    X_true = np.load(PATH+'/data/SPAHM_a_H2O/X_H2O.npy', allow_pickle=True)
    # the next two lines deviate from the common template
    a = X[0]
    assert(X.shape == np.array(X_true[0], ndmin=2).shape)
    for a_true in X_true:
        if a[0] == a_true[0]:                       # atom type
            assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations


def test_water_single_element_short():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-short', auxbasis='ccpvdzjkfit', only_z=['O'])
    X = np.array([(z,np.trim_zeros(v)) for z,v in X], dtype=object) ## trimming is necessary to get the short-version vector !
    X_true = np.load(PATH+'/data/SPAHM_a_H2O/X_H2O_lowdin-short.npy', allow_pickle=True)
    a = X[0]
    assert(X.shape == np.array(X_true[0], ndmin=2).shape)
    for a_true in X_true:
        if a[0] == a_true[0]:                       # atom type
            assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations


def test_water_single_element_SAD():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=0, spin=0)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=None, with_symbols=True,
                         xc = 'hf', model='sad-diff', auxbasis='ccpvdzjkfit', only_z=['O'])
    X = np.array([(z,np.trim_zeros(v)) for z,v in X], dtype=object) ## trimming is necessary to get the short-version vector !
    X_true = np.load(PATH+'/data/SPAHM_a_H2O/X_H2O_SAD.npy', allow_pickle=True)
    a = X[0]
    assert(X.shape == np.array(X_true[0], ndmin=2).shape)
    for a_true in X_true:
        if a[0] == a_true[0]:                       # atom type
            assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations


if __name__ == '__main__':
    test_water()
    test_water_alternate()
    test_water_lowdinshort()
    test_water_lowdinshortx()
    test_water_lowdinlong()
    test_water_SAD_guess_close_shell()
    test_water_SAD_guess_open_shell()
    test_water_single_element()
    test_water_single_element_short()
    test_water_mr21()
    test_water_single_element_SAD()
