#!/usr/bin/env python3

import os
import numpy as np
from pyscf import gto, data
from qstack import compound, spahm


def build_mol(mol, r):
    r = r.reshape((-1,3))
    mymol = gto.Mole()
    mymol.charge = mol.charge
    mymol.spin   = mol.spin
    mymol.ecp    = mol.ecp
    mymol.atom = [ (mol.atom_symbol(i), r[i]) for i in range(mol.natm)]
    mymol.basis = mol.basis
    mymol.build()
    return mymol


def grad_num(func, mol, guess, eps=1e-4):
    r = mol.atom_coords(unit='ang').flatten()
    return derivatives_num(r, func, mol, guess, eps)*data.nist.BOHR


def derivatives_num(r, func, mol, guess, eps=1e-4):
    g = []
    for i in range(len(r)):
        u   = np.eye(1, len(r), i).flatten()  # unit vector || ith dimension
        e1  = func(r+eps*u, mol, guess)
        e2  = func(r-eps*u, mol, guess)
        e11 = func(r+2*eps*u, mol, guess)
        e22 = func(r-2*eps*u, mol, guess)
        g.append((8.0*e1-8.0*e2 + e22-e11) / (12.0*eps))
    return np.array(g)


def test_spahm_ev_grad():
    def spahm_ev(r, mol, guess):
        mymol = build_mol(mol, r)
        e, _ = spahm.compute_spahm.get_guess_orbitals(mymol, guess[0])
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', 'def2svp', charge=0, spin=0)
    guess = spahm.guesses.get_guess_g('lb')
    agrad = spahm.compute_spahm.get_guess_orbitals_grad(mol, guess)[1].reshape(-1, mol.natm*3)
    ngrad = grad_num(spahm_ev, mol, guess).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


def test_spahm_re_grad():
    def spahm_re(r, mol, guess_in):
        mymol = build_mol(mol, r)
        e = spahm.compute_spahm.get_spahm_representation(mymol, guess_in)
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', 'def2svp', charge=1, spin=1)
    guess = 'lb'
    agrad = spahm.compute_spahm.get_spahm_representation_grad(mol, guess)[1].reshape(-1, mol.natm*3)
    ngrad = grad_num(spahm_re, mol, guess).reshape(mol.natm*3, -1).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


def test_spahm_ev_grad_ecp():
    def spahm_ev(r, mol, guess):
        mymol = build_mol(mol, r)
        e, _ = spahm.compute_spahm.get_guess_orbitals(mymol, guess[0])
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2Te.xyz', 'minao', charge=0, spin=0, ecp='def2-svp')
    guess = spahm.guesses.get_guess_g('lb')
    agrad = spahm.compute_spahm.get_guess_orbitals_grad(mol, guess)[1].reshape(-1, mol.natm*3)
    ngrad = grad_num(spahm_ev, mol, guess).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


def test_spahm_ev_grad_field():
    def spahm_ev(r, mol, guess):
        mymol = build_mol(mol, r)
        e, _ = spahm.compute_spahm.get_guess_orbitals(mymol, guess[0], field=field)
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', 'def2svp', charge=0, spin=0)
    field = np.array((0.01, 0.01, 0.01))
    guess = spahm.guesses.get_guess_g('lb')
    agrad = spahm.compute_spahm.get_guess_orbitals_grad(mol, guess, field=field)[1].reshape(-1, mol.natm*3)
    ngrad = grad_num(spahm_ev, mol, guess).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


def test_spahm_re_grad_field():
    # test spahm derivatives wrt atom positions in external field
    def spahm_re(r, mol, guess_in):
        mymol = build_mol(mol, r)
        e = spahm.compute_spahm.get_spahm_representation(mymol, guess_in, field=field)
        return e
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', 'def2svp', charge=1, spin=1)
    field = np.array((0.01, 0.01, 0.01))
    guess = 'lb'
    agrad = spahm.compute_spahm.get_spahm_representation_grad(mol, guess, field=field)[1].reshape(-1, mol.natm*3)
    ngrad = grad_num(spahm_re, mol, guess).reshape(mol.natm*3, -1).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


def test_spahm_re_field_grad():
    # test spahm derivatives wrt external field
    def spahm_re(field, mol, guess_in):
        return spahm.compute_spahm.get_spahm_representation(mol, guess_in, field=field)
    path  = os.path.dirname(os.path.realpath(__file__))
    mol   = compound.xyz_to_mol(path+'/data/H2O_dist_rot.xyz', 'def2svp', charge=0, spin=0)
    field = np.array((0.02, 0.02, 0.02))
    guess = 'lb'
    agrad = spahm.compute_spahm.get_spahm_representation_grad(mol, guess, field=field)[2].reshape(-1, 3)
    ngrad = derivatives_num(field, spahm_re, mol, guess).reshape(3, -1).T
    for g1, g2 in zip(ngrad, agrad, strict=True):
        assert(np.linalg.norm(g1-g2)<1e-6)


if __name__ == '__main__':
    test_spahm_ev_grad()
    test_spahm_re_grad()
    test_spahm_ev_grad_ecp()
    test_spahm_ev_grad_field()
    test_spahm_re_grad_field()
    test_spahm_re_field_grad()
