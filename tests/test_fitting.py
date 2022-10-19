#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.fields.decomposition import decompose


def test_fitting():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c0 = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    auxmol, c = decompose(mol, dm, 'cc-pvdz jkfit')
    assert(np.linalg.norm(c-c0)<1e-10)

if __name__ == '__main__':
    test_fitting()
