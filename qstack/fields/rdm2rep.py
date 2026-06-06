#!/usr/bin/env python3

import numpy as np
import pyscf.df



def _rdm2_repulse_point_formula(mol, rdm2, mask, aos, points, out=None):
    fakemol = pyscf.gto.fakemol_for_charges(points)
    eri3 = pyscf.df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij')

    return np.einsum("klq,q,qp,pk,pl->p", rdm2, mask, eri3, aos.conj(), aos, optimize=True, out=out)

def _rdm2_repulse_onepoint_formula(mol, rdm2, mask, aos, point, _cintopt, _batch_size=None):

    mol.set_rinv_origin(point)
    eri3 = mol.intor('int1e_rinv')
    eri3 = pyscf.lib.pack_tril(eri3)

    out = 0.0
    for ao_begin in range(0, mol.nao, _batch_size):
        ao_end = min(ao_begin+_batch_size, mol.nao)
        out += np.einsum("klq,q,q,k,l->", rdm2[:,ao_begin:ao_end,:], mask, eri3, aos.conj(), aos[ao_begin:ao_end])
    return out


def compute_rdm2_repulse_points(mol, rdm2, MAX_SZ=5*1024**3, grid_level=3):
    atm_grids = pyscf.dft.grid.gen_atomic_grids(mol, level=grid_level)
    grid, weights = pyscf.dft.grid.gen_partition(mol,atm_grids)

    floats_per_point = (mol.nao**3)*(mol.nao+1)

    max_points = MAX_SZ // (floats_per_point * np.dtype(rdm2.dtype).itemsize)
    rep = np.zeros_like(rdm2, shape=len(weights))
   
    rdm2 = pyscf.lib.pack_tril(rdm2.reshape(-1,mol.nao,mol.nao)).reshape(mol.nao,mol.nao,-1)
    mask = pyscf.lib.pack_tril(2*np.ones((mol.nao,mol.nao), dtype=rdm2.dtype) - np.eye(mol.nao, dtype=rdm2.dtype))
    aos = mol.eval_ao("GTOval_sph", grid)

    if max_points<2:
        floats_per_subpoint = (mol.nao**2)*(mol.nao+1)
        max_subpoints = MAX_SZ // (floats_per_subpoint * np.dtype(rdm2.dtype).itemsize)
        cintopt = pyscf.gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int1e_rinv')

        for point_i in range(len(weights)):
            rep[point_i] = _rdm2_repulse_onepoint_formula(mol, rdm2, mask, aos[point_i], grid[point_i], cintopt, max_subpoints) 
    else:
        for point_begin in range(0, len(weights), max_points):
            point_end = min(point_begin+max_points, len(weights))
            _rdm2_repulse_point_formula(mol, rdm2, mask, aos[point_begin:point_end], grid[point_begin:point_end], rep[point_begin:point_end])
    return grid, rep, weights




