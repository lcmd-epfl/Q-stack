#!/usr/bin/env python3

import os
import numpy as np
import qstack
from pyscf import dft


def test_tests():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)

    xc = 'HF'
    if mol.multiplicity == 1:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.kernel()
    #dm = mf.make_rdm1()
