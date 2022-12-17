#!/usr/bin/env python3

import os
import numpy as np
from pyscf import gto, data
from qstack import compound, spahm

np.set_printoptions(formatter={'all':lambda x: '% .4f'%x})

def test_spahm_grad():
    def grad_num(R, eps=1e-4):
        r = R.flatten()
        g = []
        for i,ri in enumerate(r):
            r[i] = ri+eps
            e1 = myfunc(r)
            r[i] = ri-eps
            e2 = myfunc(r)
            r[i] = ri
            g.append((e1-e2)*0.5/eps)
        return np.array(g)*data.nist.BOHR

    def myfunc(r):
        r = r.reshape((-1,3))
        mymol = gto.Mole()
        mymol.atom = [ (mol.atom_symbol(i), r[i]) for i in range(mol.natm)]
        mymol.basis = mol.basis
        mymol.build()
        e, c = spahm.compute_spahm.get_guess_orbitals(mymol, guess[0])
        return e

    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O_dist.xyz', 'def2svp', charge=0, spin=0)

    r0 = mol.atom_coords(unit='ang')
    guess = spahm.guesses.get_guess_g('lb')
    agrad = spahm.compute_spahm.get_guess_orbitals_grad(mol, guess)
    ngrad = grad_num(r0)

    for ind in range(mol.nao):
        g1 = ngrad[:,ind]
        g2 = agrad[ind].flatten()
        assert(np.linalg.norm(g1-g2)<1e-6)

if __name__ == '__main__':
    test_spahm_grad()
