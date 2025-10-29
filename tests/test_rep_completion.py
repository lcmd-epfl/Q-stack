#!/usr/bin/env python3

import os
import numpy as np
from qstack.spahm.rho.rep_completion import fromr1tor2

PATH = os.path.dirname(os.path.realpath(__file__))

def test_tranform_atom():
    aux_basis_set = "ccpvdzjkfit"
    set1_species = ["C", "H", "O"]
    X1_set1 = np.load(os.path.join(PATH, "data/test_CH3OH_default_saphm-a.npy"), allow_pickle=True)
    set2_species = ["H", "O"]
    X2_set2 = np.load(os.path.join(PATH, "data/test_H2O_default_saphm-a.npy"), allow_pickle=True)
    X2_set1 = fromr1tor2(X2_set2, set2_species, set1_species, aux_basis=aux_basis_set)
    assert(len(X1_set1[0,1]) == len(X2_set1[0,1])) # test the final vector-length
    assert(all([np.isclose(np.linalg.norm(x1),np.linalg.norm(x2)) for x1,x2 in zip(X2_set2[:,1], X2_set1[:,1])])) # test that atomic-reps have not changed 

#def test_transform_bond():


