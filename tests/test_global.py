import os
import numpy as np
from qstack.regression import kernel, kernel_utils



def test_avg_kernel():

    path = os.path.dirname(os.path.realpath(__file__))
    X_dir = os.path.join(path, 'data/SPAHM_a_H2O/')
    mollist = [os.path.join(X_dir, s) for s in ['X_H2O.npy', 'X_rotated_H2O.npy', 'X_H2O_dist.npy']]
    mols = [np.load(f, allow_pickle=True) for f in mollist]
    K = kernel.kernel(mols, akernel='L', gkernel='avg', sigma=1.0)

    true_K = np.array(  [[1.       ,  1.       ,  0.79179528], \
                        [1.        , 1.        , 0.79179528] , \
                        [0.79179528, 0.79179528, 1.        ]])


    assert(K.shape == (3,3))
    assert(np.abs(np.sum(K-true_K)) < 1e-05)

def test_rem_kernel():
    path = os.path.dirname(os.path.realpath(__file__))
    X_dir = os.path.join(path, 'data/SPAHM_a_H2O/')
    mollist = [os.path.join(X_dir, s) for s in ['X_H2O.npy', 'X_rotated_H2O.npy', 'X_H2O_dist.npy']]
    mols = [np.load(f, allow_pickle=True) for f in mollist]
    K = kernel.kernel(mols, akernel='L', gkernel='rem', sigma=1.0, gdict={'alpha':1.0, 'normalize':1})

    true_K = np.array(  [[1.      , 0.6528238, 1.      ], \
                        [0.6528238,1.        ,0.6528238], \
                        [1.       ,0.6528238 ,1.       ]])

    assert(K.shape == (3,3))
    assert(np.abs(np.sum(K-true_K)) < 1e-05)

