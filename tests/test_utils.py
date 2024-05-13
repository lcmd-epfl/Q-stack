#!/usr/bin/env python3

import os
import numpy as np
import qstack.spahm.rho.utils as ut


def test_load_rep_from_list():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2list = os.path.join(path, 'data/SPAHM_a_H2O/')
    Xarray, symbols = ut.load_reps(paths2list+'reps_list.txt', from_list=True, single=False, \
            with_labels=True, local=True, summ=False, printlevel=0, \
            srcdir=paths2list)
    assert(Xarray.shape == (9,207))
    assert(len(symbols) == 9)

def test_load_reps():
    path = os.path.dirname(os.path.realpath(__file__))

    paths2X = os.path.join(path, 'data/SPAHM_a_H2O/X_H2O.npy')
    X, symbols = ut.load_reps(paths2X, from_list=False, single=True, \
            with_labels=True, local=True, summ=False, printlevel=0)
    assert(X.shape == (3,207))
    assert(len(symbols) == 3)
