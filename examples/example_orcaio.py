#!/usr/bin/env python3

import os
import argparse
import numpy as np
from qstack import compound, fields
from qstack.io import orca


def example_orca_reader():
    parser = argparse.ArgumentParser(description='Example of ORCA reader')
    parser.add_argument('--name', type=str, default='AJALIH', choices=['AJALIH', 'CEHZOF'], help='job name')
    args = parser.parse_args()

    path = os.path.dirname(os.path.realpath(__file__))
    basis = 'def2-TZVP'
    version = 502

    name = args.name
    xyz = f'{path}/data/{name}/{name}.xyz'
    mol = compound.xyz_to_mol(xyz, basis, ecp=basis, parse_comment=True)
    name_long = f'{name}_{mol.multiplicity}_SPE'
    gbw = f'{path}/data/{name}/{name}_{mol.multiplicity}_SPE.gbw'
    openshell = mol.multiplicity > 1

    S = mol.intor('int1e_ovlp_sph')
    C, _e, occ = orca.read_gbw(mol, gbw)
    D1 = fields.dm.make_rdm1(np.squeeze(C), np.squeeze(occ))
    D2 = orca.read_density(mol, name_long, reorder_dest='pyscf',
                             version=version, openshell=openshell, directory=f'{path}/data/{name}')

    if openshell:
        Da, Db = D1
        print()
        print(mol.nelec)
        print(np.trace(Da@S), np.trace(Db@S))
        Dp, Ds = D2
        print()
        print(mol.nelectron, mol.multiplicity-1)
        print(np.trace(Dp@S), np.trace(Ds@S))
    else:
        print()
        print(mol.nelectron)
        print(np.trace(D1@S))
        print(np.trace(D2@S))


if __name__ == '__main__':
    example_orca_reader()
