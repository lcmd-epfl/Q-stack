#!/usr/bin/env python3

import os
import itertools
import numpy as np
from qstack.spahm.rho import utils, compute_rho_spahm as bond

PATH = os.path.dirname(os.path.realpath(__file__))


def underlying_test(X, truepath):
    true_file = PATH + truepath
    X_true = np.load(true_file)
    assert (X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true, strict=True):
        assert (np.linalg.norm(Xa-Xa_true) < 1e-8)  # evaluating representation diff as norm (threshold = 1e-8)


def test_water():
    xyz_in = PATH+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    X = bond.get_repr("bond", mols, [xyz_in], 'LB', spin=[0], with_symbols=False, same_basis=False)
    underlying_test(X, '/data/H2O_spahm_b.npy_alpha_beta.npy')


def test_water_onlym0():
    xyz_in = PATH+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], spin=None, charge=None, basis='minao')
    X = bond.get_repr("bond", mols, [xyz_in], 'LB', spin=[0], with_symbols=False, same_basis=False, only_m0=True)
    underlying_test(X, '/data/H2O_spahm_b_onlym0.npy')


def test_water_closed():
    xyz_in = PATH+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [None], [0], 'minao')
    X = bond.get_repr("bond", mols, [xyz_in], 'LB', spin=[None], with_symbols=False, same_basis=False)
    underlying_test(X, '/data/H2O_spahm_b.npy')


def test_water_O_only():
    xyz_in = PATH+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.spahm_a_b("bond", mols, dms, only_z=['O'])
    X = np.squeeze(X)  # contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X)  # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main

    X_true = np.load(PATH+'/data/H2O_spahm_b.npy_alpha_beta.npy')
    X_true = X_true[0]  # this line makes it incompatible with a call to underlying_test()
    assert (X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true, strict=True):
        assert (np.linalg.norm(Xa-Xa_true) < 1e-8)  # evaluating representation diff as norm (threshold = 1e-8)


def test_water_same_basis():
    xyz_in = PATH+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.spahm_a_b("bond", mols, dms, same_basis=True)
    X = np.squeeze(X)  # contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X)  # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    underlying_test(X, '/data/H2O_spahm_b_CCbas.npy_alpha_beta.npy')


def test_ecp():
    xyz_in = PATH+'/data/I2.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao', ecp='def2-svp')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.spahm_a_b("bond", mols, dms, same_basis=True)
    X = np.squeeze(X)  # contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X)  # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main

    underlying_test(X, '/data/I2_spahm-b_minao-def2-svp_alpha-beta.npy')


def test_repr_shapes():
    xyz_in = [PATH+'/data/H2O.xyz', PATH+'/data/HO_spinline.xyz']
    mols = utils.load_mols(xyz_in, [0,-1], [0,0], 'ccpvdz')

    for with_symbols, split, merge in itertools.product([False,True],repeat=3):
        X = bond.get_repr("bond", mols, xyz_in, 'LB', spin=None, with_symbols=with_symbols, merge=merge, split=split, same_basis=False)

        if split:
            assert X.ndim == 2 - int(merge)  # shape of (Nmods[optional], Nmols). each element is another array
        else:
            assert X.ndim == 3 - int(merge)  # shape of (Nmods[optional], NatomsTot, 2 OR Nfeatures)

        # Nmods
        if not merge:
            assert X.shape[0] == 1
            X = X.reshape(X.shape[1:])

        # Nmols,N_atoms or NatomsTot
        if split:
            assert X.shape[0] == 2
            assert X[0].shape[0] == 3
            assert X[1].shape[0] == 2
            X = np.concatenate(list(X), axis=0)
        assert X.shape[0] == 5
        assert X.ndim == 2

        if with_symbols:
            assert X.shape[1] == 2
            X = np.asarray(list(X[:,1]), dtype=float)
        assert X.shape[-1] > 100


def test_from_list():
    path2list = PATH+'/data/list_water.txt'
    path2spins = PATH+'/data/list_water_spins.txt'
    path2charges = PATH+'/data/list_water_charges.txt'
    xyzlist = utils.get_xyzlist(path2list)
    spins = utils.get_chsp(path2spins, len(xyzlist))
    charges = utils.get_chsp(path2charges, len(xyzlist))
    mols = utils.load_mols(xyzlist, charges, spins, 'minao', srcdir=PATH+"/data/")
    spahm_b = bond.get_repr("bond", mols, xyzlist, 'LB', spin=spins, same_basis=True)
    Xtrue = np.load(PATH+'/data/list_H2O_spahm-b_minao_LB_alpha-beta.npy')
    assert (np.allclose(Xtrue, spahm_b))


if __name__ == '__main__':
    test_water()
    test_water_closed()
    test_water_O_only()
    test_water_same_basis()
    test_water_onlym0()
    test_ecp()
    test_repr_shapes()
    test_from_list()
