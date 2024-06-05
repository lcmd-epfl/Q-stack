#!/usr/bin/env python3

import os
import numpy as np
from pyscf.dft import numint
from qstack import compound
from qstack.fields.dm import make_grid_for_rho
from qstack.fields.dori import dori, dori_on_grid, compute_rho


def grad_num(func, grid, eps=1e-4, **kwargs):
    g = np.zeros_like(grid)
    for i, r in enumerate(grid):
        for j in range(3):
            u   = np.eye(1, len(r), j)  # unit vector || jth dimension
            e1  = func(r+eps*u, **kwargs)
            e2  = func(r-eps*u, **kwargs)
            e11 = func(r+2*eps*u, **kwargs)
            e22 = func(r-2*eps*u, **kwargs)
            g[i,j] = (8.0*e1-8.0*e2 + e22-e11) / (12.0*eps)
    return g


def test_derivatives():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')

    grid = make_grid_for_rho(mol, grid_level=3)
    ao_value = numint.eval_ao(mol, grid.coords, deriv=2)
    rho, drho_dr, d2rho_dr2 = compute_rho(mol, grid.coords, dm=dm)
    rho0 = numint.eval_rho(mol, ao_value, dm, xctype='MGGA')
    assert 1e-10 > np.linalg.norm(rho-rho0[0])
    assert 1e-10 > np.linalg.norm(drho_dr-rho0[1:4])
    assert 1e-7  > np.linalg.norm(sum(d2rho_dr2[i,i] for i in range(3))-rho0[4])


def test_dori_deriv():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'cc-pvdz', charge=0, spin=0)
    dm = np.load(path+'/data/H2O_dist.ccpvdz.dm.npy')

    grid = np.array([[0.5,0.2,1.4], [0.9,0,0], [2,2,2], *[x[1] for x in mol._atom]])
    dori_anal, rho, _ = dori_on_grid(mol, grid, dm=dm)

    def compute_k2(coords, mol=None, dm=None):
        rho, drho_dr, d2rho_dr2 = compute_rho(mol, coords, dm=dm)
        k = drho_dr / rho
        return np.einsum('xi,xi->i', k, k)
    k2 = compute_k2(grid, mol=mol, dm=dm)
    dk2_dr = grad_num(compute_k2, grid, eps=1e-4, mol=mol, dm=dm)
    dk2_dr_square = np.einsum('ix,ix->i', dk2_dr, dk2_dr)
    theta = dk2_dr_square / k2**3
    dori_num = theta / (theta + 1.0)
    assert np.allclose(dori_anal, dori_num)


def test_dori():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/dori/H6CN____monA_0012.xyz', 'sto3g', charge=1, spin=0)
    dm = np.load(path+'/data/dori/H6CN____monA_0012.hf.sto3g.dm.npy')
    dori1, rho1, s2rho1, _, _ = dori(mol, dm=dm, grid_type='cube', resolution=0.5)
    dori0, rho0, s2rho0 = np.loadtxt(path+'/data/dori/H6CN____monA_0012.dori.dat').T
    dori0 = 4.0 * dori0/(1-dori0) / (4.0 * dori0/(1-dori0) + 1) # TODO the C code gives theta 4 times smaller than this code
    idx = np.where(rho0>1e-4)
    assert np.all(abs(s2rho0[idx]-s2rho1[idx])<1e-4) #TODO new test data
    assert np.all(abs(rho0-rho1)<1e-4)
    assert np.all(abs(dori0-dori1)<5e-5)


def test_dori_num():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/dori/H6CN____monA_0012.xyz', 'sto3g', charge=1, spin=0)
    dm = np.load(path+'/data/dori/H6CN____monA_0012.hf.sto3g.dm.npy')
    dori1, _, _, _, _ = dori(mol, dm=dm, grid_type='cube', resolution=0.5, alg='a', mem=1/1024)
    dori2, _, _, _, _ = dori(mol, dm=dm, grid_type='cube', resolution=0.5, alg='n', mem=1/512)
    assert np.all(abs(dori2-dori1)<1e-11)


def test_dori_df():
    path = os.path.dirname(os.path.realpath(__file__))
    mol  = compound.xyz_to_mol(path+'/data/dori/H6CN____monA_0012.xyz', 'cc-pvdz jkfit', charge=1, spin=0)
    c = np.load(path+'/data/dori/H6CN____monA_0012.hf.sto3g.ccpvdzjkfit.c.npy')
    dori1, rho1, s2rho1, _, _ = dori(mol, c=c, grid_type='cube', resolution=0.5)
    dori0, rho0, s2rho0 = np.load(path+'/data/dori/H6CN____monA_0012.hf.sto3g.ccpvdzjkfit.dori.npy')
    assert np.allclose(dori0, dori1)
    idx = np.where(rho0>1e-4)
    assert np.allclose(s2rho0[idx], s2rho1[idx]) #TODO new test data


if __name__ == '__main__':
    test_derivatives()
    test_dori_deriv()
    test_dori()
    test_dori_df()
    test_dori_num()
