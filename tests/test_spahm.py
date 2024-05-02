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


if __name__ == '__main__':
    test_spahm_huckel()
    test_spahm_LB()
    test_spahm_LB_ecp()
