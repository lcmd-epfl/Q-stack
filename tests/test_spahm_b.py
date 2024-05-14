import os
import numpy as np
from qstack.spahm.rho import bond, utils
from qstack import compound

def test_water():
    path = os.path.dirname(os.path.realpath(__file__))
    xyz_in = path+'/data/H2O.xyz'
    mols = utils.load_mols([xyz_in], [0], [0], 'minao')
    dms = utils.mols_guess(mols, [xyz_in], 'LB', spin=[0])
    X = bond.bond(mols, dms, spin=[0])
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
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
    X = bond.bond(mols, dms, spin=[0], only_z=['O'])
    assert(X.shape == (2, 1, 554))
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    X = np.squeeze(X) #contains a single elements but has shape (1,Nfeat)
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
    X = bond.bond(mols, dms, spin=[0], same_basis=True)
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
    X = bond.bond(mols, dms, spin=[None], same_basis=True)
    X = np.hstack(X) # merging alpha-beta components for spin unrestricted representation #TODO: should be included into function not in main
    true_file = path+'/data/I2_spahm-b_minao-def2-svp_alpha-beta.npy'
    X_true = np.load(true_file)
    assert(X_true.shape == X.shape)
    for Xa, Xa_true in zip(X, X_true):
        assert(np.linalg.norm(Xa-Xa_true) < 1e-8) # evaluating representation diff as norm (threshold = 1e-8)


if __name__ == '__main__':
    test_water()

