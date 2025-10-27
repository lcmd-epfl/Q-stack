#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm import compute_spahm


def test_spahm_huckel():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    R = compute_spahm.get_spahm_representation(mol, 'huckel')
    true_R = np.array([[-20.78722617,  -1.29750913,  -0.51773954,  -0.4322361 , -0.40740531],
                       [-20.78722617,  -1.29750913,  -0.51773954,  -0.4322361 , -0.40740531]])
    assert(R.shape == (2,5))
    assert(np.abs(np.sum(R-true_R)) < 1e-05)


def test_spahm_LB():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=1, spin=1)
    R = compute_spahm.get_spahm_representation(mol, 'lb')
    true_R = np.array( [[-18.80209878,  -1.28107468,  -0.79949967,  -0.63587071,  -0.57481672],
                        [-18.80209878,  -1.28107468,  -0.79949967,  -0.63587071,   0.        ]])
    assert(R.shape == (2,5))
    assert(np.abs(np.sum(R-true_R)) < 1e-05)


def test_spahm_LB_ecp():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2Te.xyz', 'minao',
                              ecp='def2-svp', charge=0, spin=0)
    R = compute_spahm.get_spahm_representation(mol, 'lb')[0]
    true_R = np.array([-5.68297474, -3.91180966, -3.91176332, -3.90721427, -1.22967252, -1.22469672,
                       -1.22145412, -1.2210437 , -1.22099792, -0.43285024, -0.20943343, -0.15915716,
                       -0.07260264])
    assert np.allclose(R, true_R)

    mol2 = compound.xyz_to_mol(path+'/data/I2.xyz', 'minao',
                              ecp='def2-svp', charge=0, spin=0)
    R2 = compute_spahm.get_spahm_representation(mol2, 'lb')[0]
    true_R2 = np.array([-6.2945979,  -6.29443973, -4.40192682, -4.40191206, -4.39603818, -4.39603818,
                       -4.39595727, -4.39595727 ,-1.57658748 ,-1.57157299 ,-1.57157299 ,-1.57108593 ,
                       -1.57002121, -1.57002121 ,-1.5654719  ,-1.5654719  ,-1.56533001 ,-1.56533001 ,
                       -0.53949273, -0.43239533 ,-0.19728471 ,-0.1501003  ,-0.1501003  ,-0.07957994 ,
                       -0.07957994])
    assert np.allclose(R2, true_R2)


def test_spahm_LB_field():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp')
    R = compute_spahm.get_spahm_representation(mol, 'lb', field=(0.01,0.01,0.01))
    true_R = np.array([-18.26790464,  -0.7890498,   -0.32432933,  -0.17412611,  -0.10335613])
    assert np.allclose(R[0], R[1])
    assert np.allclose(true_R, R[0])


def test_generate_reps():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/mols/')
    xyzlist = [os.path.join(path,s) for s in sorted(os.listdir(path)) if ".xyz" in s]
    mols = [compound.xyz_to_mol(f, basis='minao', charge=0, spin=0) for f in xyzlist]
    xmols = [compute_spahm.get_spahm_representation(mol, 'lb')[0] for mol in mols]
    maxlen = max([len(x) for x in xmols])
    X = np.array([np.pad(x, pad_width=(0,maxlen-len(x)), constant_values=0) for x in xmols])
    Xtrue = np.load(os.path.join(path, 'X_lb.npy'))
    assert(np.allclose(X, Xtrue))


if __name__ == '__main__':
    test_spahm_huckel()
    test_spahm_LB()
    test_spahm_LB_ecp()
    test_spahm_LB_field()
    test_generate_reps()
