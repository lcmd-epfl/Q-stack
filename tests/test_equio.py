#!/usr/bin/env python3

import os
import tempfile, filecmp
import numpy as np
from qstack import compound, fields, equio
import metatensor


def test_equio_vector():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz jkfit', charge=0, spin=0)
    c = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    ctensor = equio.array_to_tensormap(mol, c)
    tmpfile = tempfile.mktemp() + '.mts'
    metatensor.save(tmpfile, ctensor)
    if not os.path.isfile(tmpfile):
        # different versions of metatensor impose different file extensions while saving
        # (.mts recently, .npz prior)
        tmpfile += ".npz"
    assert(filecmp.cmp(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.mts', tmpfile))
    c1 = equio.tensormap_to_array(mol, ctensor)
    assert(np.linalg.norm(c-c1)==0)

def test_equio_matrix():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    dtensor = equio.array_to_tensormap(mol, dm)
    tmpfile = tempfile.mktemp() + '.mts'
    metatensor.save(tmpfile, dtensor)
    if not os.path.isfile(tmpfile):
        tmpfile += ".npz"  # see comment in `test_equio_vector()`
    assert(filecmp.cmp(path+'/data/H2O_dist.ccpvdz.dm.mts', tmpfile))
    dm1 = equio.tensormap_to_array(mol, dtensor)
    assert(np.linalg.norm(dm-dm1)==0)

def test_equio_joinsplit():
    path = os.path.dirname(os.path.realpath(__file__))
    mol1 = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz jkfit', charge=0, spin=0)
    mol2 = compound.xyz_to_mol(path+'/data/CH3OH.xyz', 'cc-pvdz jkfit', charge=0, spin=0)
    c1 = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')
    c2 = np.load(path+'/data/CH3OH.ccpvdz.ccpvdzjkfit.npy')
    ctensor1 = equio.array_to_tensormap(mol1, c1)
    ctensor2 = equio.array_to_tensormap(mol2, c2)
    ctensor_big = equio.join([ctensor1, ctensor2])

    tmpfile = tempfile.mktemp() + '.mts'
    metatensor.save(tmpfile, ctensor_big)
    if not os.path.isfile(tmpfile):
        tmpfile += ".npz"  # see comment in `test_equio_vector()`
    assert(filecmp.cmp(path+'/data/H2O_dist_CH3OH.ccpvdz.ccpvdzjkfit.mts', tmpfile))

    ctensors = equio.split(ctensor_big)
    c11, c22 = [equio.tensormap_to_array(mol, t) for mol,t in zip([mol1,mol2], ctensors)]
    assert(np.linalg.norm(c11-c1)==0)
    assert(np.linalg.norm(c22-c2)==0)


if __name__ == '__main__':
    test_equio_vector()
    test_equio_matrix()
    test_equio_joinsplit()
