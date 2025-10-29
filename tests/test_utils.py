#!/usr/bin/env python3

import os
import tempfile
import numpy as np
import qstack.spahm.rho.utils as ut
import qstack.spahm.rho.atom as atom
import qstack.compound as compound


def test_load_rep_from_list():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2list = os.path.join(path, 'data/SPAHM_a_H2O/')
    Xarray, symbols = ut.load_reps(paths2list+'reps_list.txt', from_list=True, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True, \
            srcdir=paths2list)
    assert(Xarray.shape == (9,207))
    assert(len(symbols) == 9)

def test_load_reps():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2X = os.path.join(path, 'data/SPAHM_a_H2O/X_H2O.npy')
    X, symbols = ut.load_reps(paths2X, from_list=False, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (3,207))
    assert(len(symbols) == 3)

def test_load_reps_nosymbols(): #throws warning and returns empty list of symbols
    path = os.path.dirname(os.path.realpath(__file__))

    paths2X = os.path.join(path, 'data/H2O_spahm_b.npy_alpha_beta.npy')
    X, symbols = ut.load_reps(paths2X, from_list=False, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (3,1108))
    assert(len(symbols) == 0)

def test_load_reps_singleatom():
    path = os.path.dirname(os.path.realpath(__file__))
    tmpfile = tempfile.mktemp()+'.npy'

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr([mol], [xyzpath], 'LB', rep_type="atom",
                        elements=["H", "O"], spin=[0], with_symbols=True,
                        model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=['O']) #requesting reps for O-atom only
    np.save(tmpfile, rep)
    X, symbols = ut.load_reps(tmpfile, from_list=False, \
            with_labels=True, local=True, sum_local=False, printlevel=0, progress=True)
    assert(X.shape == (1,414))
    assert(len(symbols) == 1)
    assert(symbols[0] == 'O')

def test_load_reps_singleatom_sum_local():
    path = os.path.dirname(os.path.realpath(__file__))
    tmpfile = tempfile.mktemp()+'.npy'

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr([mol], [xyzpath], 'LB', rep_type="atom",
                        elements=["H", "O"], spin=[0], with_symbols=True,
                        model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=['O']) #requesting reps for O-atom only
    np.save(tmpfile, rep)
    X = ut.load_reps(tmpfile, from_list=False, \
            with_labels=False, local=True, sum_local=True, printlevel=0, progress=True)
    assert(X.shape == (1,414))

def test_load_reps_singleatom_sum_local2():
    path = os.path.dirname(os.path.realpath(__file__))
    tmpfile = tempfile.mktemp()+'.npy'

    xyzpath = os.path.join(path, 'data/H2O.xyz')
    mol = compound.xyz_to_mol(xyzpath, basis="minao", charge=0, spin=0, ignore=False, unit='ANG', ecp=None)
    rep = atom.get_repr([mol], [xyzpath], 'LB', rep_type="atom",
                        elements=["H", "O"], spin=[0], with_symbols=True,
                        model='lowdin-long-x', auxbasis='ccpvdzjkfit', only_z=['O']) #requesting reps for O-atom only
    np.save(tmpfile, rep)
    X = ut.load_reps(tmpfile, from_list=False, \
            with_labels=False, local=True, sum_local=True, printlevel=0, progress=True)
    assert(X.shape == (1,414))

def test_load_mols():
    path = os.path.dirname(os.path.realpath(__file__))
    molslist = [os.path.join(path, 'data', m) for m in ['H2O.xyz','H2O_dist.xyz','rotated_H2O.xyz']]
    mols = ut.load_mols(molslist, [0]*len(molslist), [None]*len(molslist), 'minao', progress=True)
    assert(len(mols) == 3)

def test_check_data_structure():
    path = os.path.dirname(os.path.realpath(__file__))
    test_files = [
                  {'path2file': os.path.join(path, 'data', 'H2O_spahm-e_def2svp.npy'), 'is_local':False, 'is_single':True, 'is_labeled':False}, \
                  {'path2file': os.path.join(path, 'data', 'H2O_spahm_b.npy_alpha_beta.npy'), 'is_local':True, 'is_single':True, 'is_labeled':False}, \
                  {'path2file': os.path.join(path, 'data', 'SPAHM_a_H2O/X_H2O.npy'), 'is_local':True, 'is_single':True, 'is_labeled':True}, \
                  {'path2file': os.path.join(path, 'data', 'SPAHM_a_H2O/Xs_H2O_array.npy'), 'is_local':True, 'is_single':False, 'is_labeled':True} \
                  ]
    for ft in test_files:
        is_single, is_labeled = ut.check_data_struct(ft['path2file'], local = ft['is_local'])
        assert((ft['is_single'] == is_single) and (ft['is_labeled'] == is_labeled))

def test_regroup_symbols():
    path = os.path.dirname(os.path.realpath(__file__))
    filelist = os.path.join(path, "./data/list_water3.txt")
    regrouped_species = ut.regroup_symbols(filelist)
    rep_count = {"H":2, "O":1}
    print(regrouped_species)
    for z,v in regrouped_species.items():
        assert(len(v) == rep_count[z])

def test_regroup_symbols_and_trim():
    path = os.path.dirname(os.path.realpath(__file__))
    filelist = os.path.join(path, "./data/list_water_lowdin-short-padded.txt")
    regrouped_species = ut.regroup_symbols(filelist, trim_reps=True)
    #trimedlist = os.path.join(path, "./data/list_water_lowdin-short.txt") ## this is not possible because of inhomogenous array
    X_truth = np.load(path+"/data/SPAHM_a_H2O/X_H2O_lowdin-short.npy", allow_pickle=True)
    regrouped_truth = {z:[] for z in regrouped_species.keys()}
    for z,v in X_truth:
        regrouped_truth[z].append(v)
    for z in regrouped_species.keys():
        assert(np.allclose(regrouped_species[z], regrouped_truth[z]))


def main():
    test_load_mols()
    test_load_reps()
    test_load_rep_from_list()
    test_check_data_structure()
    test_load_reps_nosymbols()
    test_load_reps_singleatom()
    test_load_reps_singleatom_sum_local()
    test_load_reps_singleatom_sum_local2()
    test_regroup_symbols()


if __name__ == '__main__': main()
