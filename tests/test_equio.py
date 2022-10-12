#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound, fields, equio


def test_equio():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    dm   = fields.dm.get_converged_dm(mol, xc="pbe")
    auxmol, c = fields.decomposition.decompose(mol, dm, 'cc-pvdz jkfit')

    tensor = equio.vector_to_tensormap(auxmol, c)
    print(tensor)
    print()

    c1 = equio.tensormap_to_vector(auxmol, tensor)
    assert(np.linalg.norm(c-c1)==0)


if __name__ == '__main__':
    test_equio()
