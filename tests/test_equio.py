#!/usr/bin/env python3

import os
import tempfile, filecmp
import numpy as np
from qstack import compound, fields, equio
import equistore.io


def test_equio():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz jkfit', charge=0, spin=0)

    c = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    ctensor = equio.vector_to_tensormap(mol, c)

    tmpfile = tempfile.mktemp()
    equistore.io.save(tmpfile+'.npz', ctensor)
    assert(filecmp.cmp(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npz', tmpfile+'.npz'))

    c1 = equio.tensormap_to_vector(mol, ctensor)
    assert(np.linalg.norm(c-c1)==0)


    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    dtensor = equio.matrix_to_tensormap(mol, dm)
    print(dtensor)





if __name__ == '__main__':
    test_equio()
