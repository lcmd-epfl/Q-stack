#!/usr/bin/env python3

import os
import numpy as np
import qstack.spahm.rho.utils as ut
import qstack.spahm.rho.atom as atom
import qstack.compound as compound


def test_load_rep_from_list():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2list = os.path.join(path, 'data/SPAHM_a_H2O/')
    Xarray, symbols = ut.load_reps(paths2list+'reps_list.txt', from_list=True, single=False, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True, \
            srcdir=paths2list)
    assert(Xarray.shape == (9,207))
    assert(len(symbols) == 9)

def test_load_reps():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2X = os.path.join(path, 'data/SPAHM_a_H2O/X_H2O.npy')
    X, symbols = ut.load_reps(paths2X, from_list=False, single=True, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (3,207))
    assert(len(symbols) == 3)

def test_load_reps_nosymbols(): #throws warning and returns empty list of symbols
    path = os.path.dirname(os.path.realpath(__file__))

    paths2X = os.path.join(path, 'data/H2O_spahm_b.npy_alpha_beta.npy')
    X, symbols = ut.load_reps(paths2X, from_list=False, single=True, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (3,1108))
    assert(len(symbols) == 0)

def test_load_reps_singleatom():
    path = os.path.dirname(os.path.realpath(__file__))

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr(mol, ['H', 'O'], 0, 0, dm=None, guess="LB", only_z=['O'])
    np.save(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), rep)
    X, symbols = ut.load_reps(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), from_list=False, single=True, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (1,414))
    assert(len(symbols) == 1)
    assert(symbols[0] == 'O')

def test_load_reps_singleatom_sum_local():
    path = os.path.dirname(os.path.realpath(__file__))

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr(mol, ['H', 'O'], 0, 0, dm=None, guess="LB", only_z=['H'])
    np.save(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), rep)
    X = ut.load_reps(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), from_list=False, single=True, \
            with_labels=False, local=True, sum_local=True, printlevel=0, progress=True)
    assert(X.shape == (1,414))

def test_load_reps_singleatom_sum_local():
    path = os.path.dirname(os.path.realpath(__file__))

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr(mol, ['H', 'O'], 0, 0, dm=None, guess="LB", only_z=['O']) #since only one O checks if the sum is not run over the representations elements
    np.save(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), rep)
    X = ut.load_reps(os.path.join(path, 'data/tmp_H2O-minao-ccpvdzjkfit.npy'), from_list=False, single=True, \
            with_labels=False, local=True, sum_local=True, printlevel=0, progress=True)
    assert(X.shape == (1,414))

def test_load_mols():
    path = os.path.dirname(os.path.realpath(__file__))
    molslist = [os.path.join(path, 'data', m) for m in ['H2O.xyz','H2O_dist.xyz','rotated_H2O.xyz']]
    mols = ut.load_mols(molslist, [0]*len(molslist), [None]*len(molslist), 'minao', progress=True)
    assert(len(mols) == 3)

def main():
    test_load_mols()
    test_load_reps()
    test_load_rep_from_list()

if __name__ == '__main__': main()
