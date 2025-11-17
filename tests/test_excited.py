#!/usr/bin/env python3

import os
import numpy as np
import pyscf
from qstack import compound, fields


def test_excited():

    path = os.path.dirname(os.path.realpath(__file__))
    xyzfile = path+'/data/excited/C1-13-2-3.xyz'
    mol     = compound.xyz_to_mol(xyzfile, 'def2svp')
    coeff   = np.load(xyzfile+'.mo.npy')
    X       = np.load(xyzfile+'.X.npy')
    x_c     = np.load(xyzfile+'.st2_transition_fit.npy')
    hole_d0 = np.load(xyzfile+'.st2_dm_hole.npy')
    part_d0 = np.load(xyzfile+'.st2_dm_part.npy')
    hole_c  = np.load(xyzfile+'.st2_dm_hole_fit.npy')
    part_c  = np.load(xyzfile+'.st2_dm_part_fit.npy')

    state_id = 1
    x_ao = fields.excited.get_transition_dm(mol, X[state_id], coeff)
    dip  = fields.moments.first(mol, x_ao)
    dip0 = np.array([ 0.68927353, -2.10714637, -1.53423419])
    assert(np.allclose(dip, dip0, atol=1e-8))

    hole_d, part_d = fields.excited.get_holepart(mol, X[state_id], coeff)
    assert(np.allclose(hole_d, hole_d0, atol=1e-8))
    assert(np.allclose(part_d, part_d0, atol=1e-8))

    auxmol = compound.make_auxmol(mol, 'ccpvqz jkfit')
    dip    = fields.moments.first(auxmol, x_c)
    dip0 = np.array([-0.68919144,  2.10692116,  1.53399871])
    assert(np.allclose(dip, dip0, atol=1e-8))

    dist, hole_extent, part_extent = fields.excited.exciton_properties(mol, hole_d, part_d)
    assert(np.allclose([dist, hole_extent, part_extent], [2.59863354, 7.84850017, 5.67617426], atol=1e-7))

    dist, hole_extent, part_extent = fields.excited.exciton_properties(auxmol, hole_c, part_c)
    assert(np.allclose([dist, hole_extent, part_extent], [2.59940378, 7.8477511,  5.67541635], atol=1e-7))


def test_excited_frag():
    path = os.path.dirname(os.path.realpath(__file__))
    xyzfile = path+'/data/excited/C1-13-2-3.xyz'
    auxmol  = compound.xyz_to_mol(xyzfile, 'ccpvqz jkfit')
    hole_c  = np.load(xyzfile+'.st2_dm_hole_fit.npy')
    part_c  = np.load(xyzfile+'.st2_dm_part_fit.npy')
    fragments = compound.fragments_read(xyzfile+'.frag')
    omega_hole_atom, omega_part_atom = fields.hirshfeld.hirshfeld_charges(auxmol, [hole_c, part_c], atm_bas='def2svp', dominant=True, occupations=True, grid_level=1)
    omega_hole_frag, omega_part_frag = compound.fragment_partitioning(fragments, [omega_hole_atom, omega_part_atom], normalize=True)
    if int(pyscf.__version__.split('.')[1]) < 7:
        omega_hole_frag0 = np.array([ 4.24465477, 25.17476403,  7.80532138, 32.88857084, 29.88668899])
        omega_part_frag0 = np.array([ 1.86936435, 20.01021326, 37.31393462, 36.74049231,  4.06599547])
    else:
        omega_hole_frag0 = np.array([ 4.24698889, 25.1717958 ,  7.80455406, 32.89098877, 29.88567248])
        omega_part_frag0 = np.array([ 1.87258999, 19.98184387, 37.30712212, 36.77858748,  4.05985653])
    assert(np.linalg.norm(omega_hole_frag-omega_hole_frag0)<1e-8)
    assert(np.linalg.norm(omega_part_frag-omega_part_frag0)<1e-8)


if __name__ == '__main__':
    test_excited()
    test_excited_frag()
