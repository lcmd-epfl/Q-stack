#!/usr/bin/env python3

import os
import argparse
import numpy as np
import scipy
from pyscf import scf
from pyscf.data import elements
from qstack import compound, orcaio, fields


def dipole_moment(mol, dm):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mass = np.round(np.array(elements.MASSES)[charges], 3)
    mass_center = np.einsum('i,ix->x', mass, coords) / sum(mass)
    scf.hf.dip_moment(mol, dm, unit='au', origin=mass_center)


def find_metal_ao_idx(mol, metals):
    for iat in range(mol.natm):
        q = mol.atom_symbol(iat)
        if q in metals:
            break
    metal_ao_limits = mol.offset_ao_by_atom()[iat]
    return np.arange(metal_ao_limits[2], metal_ao_limits[3])


def print_metal_contribution(orb_idx, c, e, metals, do_checks=False):
    sqrtS = scipy.linalg.sqrtm(mol.intor('int1e_ovlp_sph'))
    center_ao_idx = find_metal_ao_idx(mol, metals)
    if do_checks:
        print(name)
    else:
        print(name, end='')
    for set_id, orb_id in orb_idx:
        ci = c[set_id,:,orb_id]
        sqrtSci = sqrtS @ ci
        fraction_S = np.linalg.norm(sqrtSci[center_ao_idx]) * 100.0
        if do_checks:
            print(set_id, orb_id, '\t', "%10.6f"%e[set_id,orb_id], '\t', "%6.1f%%" % fraction_S)
        else:
            print('\t', "%6.1f%%" % fraction_S, end='')
    if not do_checks:
        print()


def get_homo_lumo_idx(occ):
    c_homo_idx = [(i, np.where(occ[i]>0)[0][-1]) for i in range(occ.shape[0])]
    c_lumo_idx = [(i, np.where(occ[i]==0)[0][0]) for i in range(occ.shape[0])]

    e_homo = [e[i] for i in c_homo_idx]
    e_lumo = [e[i] for i in c_lumo_idx]

    homo_idx = c_homo_idx[np.argmax(e_homo)]
    lumo_idx = c_lumo_idx[np.argmin(e_lumo)]
    return homo_idx, lumo_idx


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Example of ORCA reader')
    parser.add_argument('--name', type=str, default='AJALIH', choices=['AJALIH', 'CEHZOF'], help='job name')
    parser.add_argument('--basis', type=str, default='def2-tZVP', help='basis set')
    parser.add_argument('--no-checks', dest='do_checks', action='store_false', help='disable checks')
    args = parser.parse_args()

    metals = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']

    name = args.name
    path = os.path.dirname(os.path.realpath(__file__))
    xyz = f'{path}/data/{name}/{name}.xyz'
    mol = compound.xyz_to_mol(xyz, args.basis, ecp=args.basis, parse_comment=True)
    gbw = f'{path}/data/{name}/{name}_{mol.multiplicity}_SPE.gbw'

    c, e, occ = orcaio.read_gbw(mol, gbw)
    homo_idx, lumo_idx = get_homo_lumo_idx(occ)
    print_metal_contribution([homo_idx, lumo_idx], c, e, metals, do_checks=args.do_checks)

    if args.do_checks:
        dm = fields.dm.make_rdm1(np.squeeze(c), np.squeeze(occ))
        dipole_moment(mol, dm)
        gap = e[lumo_idx]-e[homo_idx]
        print(f'{gap=:6f} au ({gap*27.2:6f} eV)')
