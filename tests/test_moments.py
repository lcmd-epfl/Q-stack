#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.fields import decomposition, moments


def test_moments():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c = decomposition.decompose(mol, dm, 'cc-pvdz')[1]

    R0 = 9.930396060748974
    R0_atom = [5.6426496,  1.88412837, 2.4036181 ]
    R1 = [ 1.53224245e-01,  1.70535989e-01, -8.51874261e-16]
    R2 = 12.352661975356678

    r0, r1, r2 = moments.r2_c(mol, c)
    assert (np.allclose(r0, R0))
    assert (np.allclose(r1, R1))
    assert (np.allclose(r2, R2))

    I0, I1, I2 = moments.r2_c(mol, None)
    assert (np.allclose(r0, I0@c))
    assert (np.allclose(r1, I1@c))
    assert (np.allclose(r2, I2@c))

    I0, I1, I2 = moments.r2_c(mol, None, per_atom=True)
    r0_atom = c @ I0
    assert (np.allclose(r0_atom, R0_atom))
    r1_atom = np.einsum('p,xpa->ax', c, I1)  # (atom, component)
    assert (np.allclose(r1_atom.sum(axis=0), R1))

    r0_atom, r1_atom, r2_atom = moments.r2_c(mol, c, per_atom=True)
    assert (np.allclose(r0_atom, R0_atom))
    assert (np.allclose(r1_atom.sum(axis=0), R1))
    assert (np.allclose(r2_atom.sum(), R2))


if __name__ == '__main__':
    test_moments()
