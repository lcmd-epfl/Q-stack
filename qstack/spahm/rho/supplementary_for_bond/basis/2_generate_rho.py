#!/usr/bin/env python3

import argparse
import numpy as np
from pyscf import gto,scf,dft
from pyscf_ext import readmol

parser = argparse.ArgumentParser(description='Generate the density to be fitted')
parser.add_argument('molecule',     metavar='molecule', type=str, help='xyz file')
parser.add_argument('dm',           metavar='dm',       type=str, help='dm file')
parser.add_argument('basis',        metavar='basis',    type=str, help='ao basis')
parser.add_argument('output',       metavar='output',   type=str, help='output file')
parser.add_argument('a1',           metavar='a1',       type=int, help='atom 1')
parser.add_argument('a2',           metavar='a2',       type=int, help='atom 2')
parser.add_argument('-g', '--grid', metavar='grid',     type=int, help='grid level', default=3)
args = parser.parse_args()

mol = readmol(args.molecule, args.basis, ignore=True)

grid = dft.gen_grid.Grids(mol)
grid.level = args.grid
grid.build()

dm = np.load(args.dm)
ao = dft.numint.eval_ao(mol, grid.coords)
rho = np.einsum('pq,ip,iq->i', dm, ao, ao)

r1 = mol.atom_coord(args.a1-1, unit='ANG')
r2 = mol.atom_coord(args.a2-1, unit='ANG')
rm = (r1+r2)*0.5
atom = "No  % f % f % f" % (rm[0], rm[1], rm[2])

np.savez(args.output, atom=atom, rho=rho, coords=grid.coords, weights=grid.weights)

