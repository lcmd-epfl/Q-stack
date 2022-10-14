#!/usr/bin/env python3

import os
from qstack import compound, fields

def test_hf_otpd():

    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
    dm = fields.dm.get_converged_dm(mol, xc="pbe")
    otpd, grid = fields.hf_otpd.hf_otpd(mol, dm, return_all=True)
    assert(abs(otpd @ grid.weights-20.190382555868)<1e-10)


if __name__ == '__main__':
    test_hf_otpd()
