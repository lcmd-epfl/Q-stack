import os
import numpy as np
from qstack.spahm.rho import bond, utils
from qstack import compound

def test_water():
    path = os.path.dirname(os.path.realpath(__file__))
    xyz_in = path+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.get_repr(mols, [xyz_in], 'LB', spin=[0], with_symbols=False, same_basis=False)
    true_file = path+'/data/H2O_spahm_b.npy_alpha_beta.npy'
    X_true = np.load(true_file)
    assert(X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true):
        assert(np.linalg.norm(Xa-Xa_true) < 1e-8) # evaluating representation diff as norm (threshold = 1e-8)

def test_water_O_only():
    path = os.path.dirname(os.path.realpath(__file__))
    xyz_in = path+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.bond(mols, dms, only_z=['O'])
    X = np.squeeze(X) #contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    true_file = path+'/data/H2O_spahm_b.npy_alpha_beta.npy'
    X_true = np.load(true_file)
    X_true = X_true[0]
    assert(X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true):
        assert(np.linalg.norm(Xa-Xa_true) < 1e-8) # evaluating representation diff as norm (threshold = 1e-8)

def test_water_same_basis():
    path = os.path.dirname(os.path.realpath(__file__))
    xyz_in = path+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.bond(mols, dms, same_basis=True)
    X = np.squeeze(X) #contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    true_file = path+'/data/H2O_spahm_b_CCbas.npy_alpha_beta.npy'
    X_true = np.load(true_file)
    assert(X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true):
        assert(np.linalg.norm(Xa-Xa_true) < 1e-8) # evaluating representation diff as norm (threshold = 1e-8)

def test_ecp():
    path = os.path.dirname(os.path.realpath(__file__))
    xyz_in = path+'/data/I2.xyz'
    mols = utils.load_mols([xyz_in], [0], [None], 'minao', ecp='def2-svp')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[None])
    X = bond.bond(mols, dms, same_basis=True)
    X = np.squeeze(X) #contains a single elements but has shape (1,Nfeat)
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    true_file = path+'/data/I2_spahm-b_minao-def2-svp_alpha-beta.npy'
    X_true = np.load(true_file)
    assert(X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true):
        assert(np.linalg.norm(Xa-Xa_true) < 1e-8) # evaluating representation diff as norm (threshold = 1e-8)

def test_from_list():
    path = os.path.dirname(os.path.realpath(__file__))
    path2list = path+'/data/list_water.txt'
    path2spins = path+'/data/list_water_spins.txt'
    path2charges = path+'/data/list_water_charges.txt'
    xyzlist = utils.get_xyzlist(path2list)
    spins = utils.get_chsp(path2spins, len(xyzlist))
    charges = utils.get_chsp(path2charges, len(xyzlist))
    mols = utils.load_mols(xyzlist, charges, spins, 'minao', srcdir=path+'/data/')
    spahm_b = bond.get_repr(mols, xyzlist, 'LB', spin=spins, same_basis=True)
    Xtrue = np.load(path+'/data/list_H2O_spahm-b_minao_LB_alpha-beta.npy')
    assert(np.allclose(Xtrue, spahm_b))


if __name__ == '__main__':
    test_water()
    test_from_list()

