#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound, fields, basis_opt


def test_hf_otpd():

    path = os.path.dirname(os.path.realpath(__file__))

    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    dm = fields.dm.get_converged_dm(mol, xc="pbe")
    otpd, grid = fields.hf_otpd.hf_otpd(mol, dm, return_all = True)
    mol_dict = {'atom': mol.atom, 'rho': otpd, 'coords': grid.coords, 'weights': grid.weights}
    g  = basis_opt.opt.optimize_basis(['H'], [path+'/data/initial/H_N0.txt', path+'/data/initial/O_N0.txt'], [mol_dict], check=True, printlvl=0)
    assert(np.all(g['diff'] < 1e-6))

    ob_good = {'H': [[0, [42.30256758622713, 1]], [0, [6.83662718701579, 1]], [0, [1.8547192742478775, 1]], [0, [0.3797283290452742, 1]], [1, [12.961663119622536, 1]], [1, [2.507400755551906, 1]], [1, [0.6648804678758861, 1]], [2, [3.482167705165484, 1]], [2, [0.6053728887614225, 1]], [3, [0.6284190712545101, 1]]]}
    ob = basis_opt.opt.optimize_basis(['H'], [path+'/data/initial/H_N0.txt'], [path+'/data/H2.ccpvtz.grid3.npz'], printlvl=2, gtol_in=1e-5)
    for [l,[a,c]], [l1,[a1,c1]] in zip(ob_good['H'], ob['H']):
        assert(abs(a-a1)<1e-5)

if __name__ == '__main__':
    test_hf_otpd()
