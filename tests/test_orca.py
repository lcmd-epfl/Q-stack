#!/usr/bin/env python3

import os
import numpy as np
import qstack


def test_orca_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)
    dm = qstack.fields.dm.get_converged_dm(mol, 'HF')
    dm = np.array([dm[0]+dm[1], dm[0]-dm[1]])

    dm400 = qstack.orcaio.read_density(mol, 'H2O.orca400', directory=path+'/data/orca/', version=400, openshell=True)
    dm421 = qstack.orcaio.read_density(mol, 'H2O.orca421', directory=path+'/data/orca/', version=421, openshell=True)
    dm504 = qstack.orcaio.read_density(mol, 'H2O.orca504', directory=path+'/data/orca/', version=504, openshell=True)

    assert(np.linalg.norm(dm-dm400)<1e-4)
    assert(np.linalg.norm(dm400-dm421)<1e-10)
    assert(np.linalg.norm(dm504-dm421)<5e-3)


if __name__ == '__main__':
    test_orca_reader()
