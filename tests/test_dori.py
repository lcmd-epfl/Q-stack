#!/usr/bin/env python3

import os
import numpy as np
from pyscf.dft import numint
from qstack import compound
from qstack.fields.dm import make_grid_for_rho, get_converged_dm
from qstack.fields.dori import dori, eval_rho


def test_derivatives():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')

    grid = make_grid_for_rho(mol, grid_level=3)
    ao_value = numint.eval_ao(mol, grid.coords, deriv=2)
    rho, drho_dr, d2rho_dr2 = eval_rho(mol, ao_value, dm)
    rho0 = numint.eval_rho(mol, ao_value, dm, xctype='MGGA')
    assert 1e-10 > np.linalg.norm(rho-rho0[0])
    assert 1e-10 > np.linalg.norm(drho_dr-rho0[1:4])
    assert 1e-7  > np.linalg.norm(sum(d2rho_dr2[i,i] for i in range(3))-rho0[4])


def test_dori():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/dori/H6CN____monA_0012.xyz', 'sto3g', charge=1, spin=0)
    dm = get_converged_dm(mol, 'HF')
    dori1, rho1, s2rho1, _, _ = dori(mol, dm=dm, grid_type='cube', resolution=0.5)
    dori0, rho0, s2rho0 = np.loadtxt(path+'/data/dori/H6CN____monA_0012.dori.dat').T
    dori0 = 4.0 * dori0/(1-dori0) / (4.0 * dori0/(1-dori0) + 1) # TODO the C code gives theta 4 times smaller than this code
    assert np.all(abs(s2rho0-s2rho1)<1e-4)
    assert np.all(abs(rho0-rho1)<1e-4)
    assert np.all(abs(dori0-dori1)<5e-5)


if __name__ == '__main__':
    test_derivatives()
    test_dori()
