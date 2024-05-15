#!/usr/bin/env python3

import os
import numpy as np
import qstack


def test_orca_density_reader():
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


def test_orca_gbw_reader():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = qstack.compound.xyz_to_mol(path+'/data/orca/H2O.xyz', 'sto3g', charge=1, spin=1)

    mf, _ = qstack.fields.dm.get_converged_mf(mol, 'HF')
    c = mf.mo_coeff
    e = mf.mo_energy
    occ = mf.mo_occ

    c504, e504, occ504 = qstack.orcaio.read_gbw(mol, path+'/data/orca/H2O.orca504.gbw')
    for s in range(c.shape[0]):
        for i in range(c.shape[-1]):
            assert 5e-3 > min(np.linalg.norm(c[s,:,i]-c504[s,:,i]), np.linalg.norm(c[s,:,i] + c504[s,:,i]))

    assert np.linalg.norm(e504-e) < 1e-5
    assert np.linalg.norm(occ504-occ) == 0.0  # fine since they contain only 0.0 and 1.0


if __name__ == '__main__':
    test_orca_density_reader()
    test_orca_gbw_reader()
