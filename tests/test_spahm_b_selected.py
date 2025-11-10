#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho.bond_selected import get_spahm_b_selected


def test_spahm_b_selected():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    fname = os.path.join(path, 'H2O.xyz')
    bondij = [(0, 1)]
    mols = [compound.xyz_to_mol(fname, basis='minao', charge=0, spin=0)]
    X = get_spahm_b_selected(mols, bondij, [fname])[0][1]
    Xtrue = np.load(os.path.join(path, 'H2O.xyz_1_2.npy'))
    assert (np.allclose(X, Xtrue))


if __name__ == '__main__':
    test_spahm_b_selected()
