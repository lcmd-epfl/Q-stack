#!/usr/bin/env python3

import os
import numpy as np
import sys

sys.path.append('/home/calvino/yannick/SPAHM-RHO/')
from DMbRep import *


def test_water(X_water):

    path = os.path.dirname(os.path.realpath(__file__))
    X_dir = os.path.join(path, 'SPAHM_a_H2O/')
    water = './H2O.xyz'
#    X_water = generate_ROHSPAHM(water, ["H", "O"], 0, None,\
#            dm=None,\
#            guess='LB',\
#            model='Lowdin-long-x',\
#            basis_set='minao',\
#            aux_basis_set='ccpvdzjkfit'\
#            )

    true_X = np.load(os.path.join(X_dir, 'X_H2O.npy'), allow_pickle=True)
    assert(X_water.shape == (3,2))
    atom_test = []
    for a, a_test in zip(X_water, true_X):
        atom_test.append(a[0] == a_test[0])                         # Tests atom type
        atom_test.extend(np.abs(a[1] - a_test[1]) < 1e-05)          # Test atom representations

    assert(all(atom_test))
    print('Test is OK.')

