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

if __name__ == '__main__':
    test_spahm_huckel()
    test_spahm_LB()
