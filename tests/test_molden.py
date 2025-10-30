#!/usr/bin/env python3

import os
import tempfile
import filecmp
import numpy as np
from qstack import compound
from qstack.fields.density2file import coeffs_to_molden


def test_molden():
    path = os.path.dirname(os.path.realpath(__file__))
    auxmol = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz jkfit', charge=0, spin=0)
    c = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    tmpfile = tempfile.mktemp() + '.molden'
    coeffs_to_molden(auxmol, c, tmpfile)
    assert(filecmp.cmp(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.molden', tmpfile))


if __name__ == '__main__':
    test_molden()
