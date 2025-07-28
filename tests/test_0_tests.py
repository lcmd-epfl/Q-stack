#!/usr/bin/env python3

import os
import numpy as np
import qstack


def test_tests():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    dm = qstack.fields.dm.get_converged_dm(mol, 'HF')
