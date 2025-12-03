#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom

PATH = os.path.dirname(os.path.realpath(__file__))


def underlying_test(true_data_relpath, X, trim=False, only_z=None):
    X_true = np.load(PATH+true_data_relpath, allow_pickle=True)
    if only_z is not None:
        X_true = X_true[np.isin(X_true[:,0], only_z)]
    assert X.shape == X_true.shape
    for (q, v), (q_true, v_true) in zip(X, X_true, strict=True):
        assert q == q_true
        if trim is True:
            v = np.trim_zeros(v)  # short-version vectors should be trimmed
        assert np.allclose(v, v_true)


def test_water():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O.npy', X)


def test_water_alternate():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH], 'LB', spin=[None], auxbasis='ccpvdzjkfit', with_symbols=True)
    underlying_test('/data/SPAHM_a_H2O/X_H2O.npy', X)


def test_water_lowdinshortx():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-short-x', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-short-x.npy', X, trim=True)


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
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-short.npy', X, trim=True)


def test_water_mr21():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='MR2021', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O_MR2021.npy', X, trim=True)


def test_water_SAD_guess_open_shell():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=1, spin=1)
    Xsad = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=[1], with_symbols=True,
                         xc='hf', model='sad-diff', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O-RC_SAD.npy', Xsad)


def test_water_SAD_guess_close_shell():
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=0, spin=0)
    Xsad = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=None, with_symbols=True,
                         xc='hf', model='sad-diff', auxbasis='ccpvdzjkfit')
    underlying_test('/data/SPAHM_a_H2O/X_H2O_SAD.npy', Xsad)


def test_water_single_element():
    only_z = ['O']
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=only_z)
    underlying_test('/data/SPAHM_a_H2O/X_H2O.npy', X, only_z=only_z)


def test_water_single_element_short():
    only_z = ['O']
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'LB',
                      elements=["H", "O"], spin=None, with_symbols=True,
                      model='lowdin-short', auxbasis='ccpvdzjkfit', only_z=only_z)
    underlying_test('/data/SPAHM_a_H2O/X_H2O_lowdin-short.npy', X, only_z=only_z)


def test_water_single_element_SAD():
    only_z = ['O']
    mol = compound.xyz_to_mol(PATH+'/data/H2O.xyz', 'sto3g', charge=0, spin=0)
    X = atom.get_repr("atom", [mol], [PATH+'/data/H2O.xyz'], 'sad',
                         elements=["H", "O"], spin=None, with_symbols=True,
                         xc='hf', model='sad-diff', auxbasis='ccpvdzjkfit', only_z=only_z)
    underlying_test('/data/SPAHM_a_H2O/X_H2O_SAD.npy', X, only_z=only_z)


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
