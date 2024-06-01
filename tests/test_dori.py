#!/usr/bin/env python3

import os
import numpy as np
from pyscf.dft import numint
from qstack import compound
from qstack.fields.dm import make_grid_for_rho
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
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')
    c  = np.load(path+'/data/H2O_dist.ccpvdz.ccpvdzjkfit.npy')

    x = dori(mol, dm, grid_type='cube')
    print(x)
    y = dori(mol, dm, grid_type='dft')
    print(y)



if __name__ == '__main__':
    test_derivatives()
    test_dori()
