#!/usr/bin/env python3

import numpy as np
import pyscf.df



def _rdm2_repulse_point_formula(mol, rdm2, points, out=None):
    orbs = mol.eval_ao("GTOval_sph", points)
    fakemol = pyscf.gto.fakemol_for_charges(points)
    eri3 = pyscf.df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij')
    eri3 = pyscf.lib.unpack_tril(eri3.T)
    return np.einsum("ijkl,pij,pk,pl->p", rdm2, eri3, orbs.conj(), orbs, optimize=True, out=out)



def compute_rdm2_repulse_points(mol, rdm2, MAX_SZ=5*1024**3, grid_level=3):
    atm_grids = pyscf.dft.grid.gen_atomic_grids(mol, level=grid_level)
    grid, weights = pyscf.dft.grid.gen_partition(mol,atm_grids)

    floats_per_point = (mol.nao**4) * 2
    print(floats_per_point)

    max_points = MAX_SZ // (floats_per_point * np.dtype(rdm2.dtype).itemsize)
    rep = np.zeros_like(rdm2, shape=len(weights))
    print(max_points, len(weights))
    
    for point_begin in range(0, len(weights), max_points):
        point_end = min(point_begin+max_points, len(weights))
        _rdm2_repulse_point_formula(mol, rdm2, grid[point_begin:point_end], rep[point_begin:point_end])
    return grid, rep, weights




