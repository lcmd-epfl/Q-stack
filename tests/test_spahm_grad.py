#!/usr/bin/env python3

import os
import numpy as np
from pyscf import gto, data
from qstack import compound, spahm

np.set_printoptions(formatter={'all':lambda x: '% .4f'%x})

def build_mol(mol, r):
    r = r.reshape((-1,3))
    mymol = gto.Mole()
    mymol.charge = mol.charge
    mymol.spin   = mol.spin
    mymol.atom = [ (mol.atom_symbol(i), r[i]) for i in range(mol.natm)]
    mymol.basis = mol.basis
    mymol.build()
    return mymol

def grad_num(func, mol, guess, eps=1e-4):
    r = mol.atom_coords(unit='ang').flatten()
    g = []
    for i,ri in enumerate(r):
        r[i] = ri+eps
        e1 = func(r, mol, guess)
        r[i] = ri-eps
        e2 = func(r, mol, guess)
        r[i] = ri
        g.append((e1-e2)*0.5/eps)
    return np.array(g)*data.nist.BOHR

def test_spahm_ev_grad():
    def myfunc(r, mol, guess):
        mymol = build_mol(mol, r)
        e, c = spahm.compute_spahm.get_guess_orbitals(mymol, guess[0])
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'def2svp', charge=0, spin=0)
    guess = spahm.guesses.get_guess_g('lb')
    agrad = spahm.compute_spahm.get_guess_orbitals_grad(mol, guess)
    ngrad = grad_num(myfunc, mol, guess)
    for g1, g2 in zip(ngrad.T, agrad.reshape(-1, 9)):
        assert(np.linalg.norm(g1-g2)<1e-6)

def test_spahm_re_grad():
    def myfunc(r, mol, guess_in):
        mymol = build_mol(mol, r)
        e = spahm.compute_spahm.get_spahm_representation(mymol, guess_in)
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'def2svp', charge=1, spin=1)
    guess = 'lb'
    agrad = spahm.compute_spahm.get_spahm_representation_grad(mol, guess)
    ngrad = grad_num(myfunc, mol, guess)
    for g1, g2 in zip(ngrad.reshape(9, -1).T, agrad.reshape(-1, 9)):
        assert(np.linalg.norm(g1-g2)<1e-6)


if __name__ == '__main__':
    test_spahm_ev_grad()
    test_spahm_re_grad()
