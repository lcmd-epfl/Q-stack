#!/usr/bin/env python3

import os
import argparse
import numpy as np
from . import utils, dmb_rep_bond as dmbb, lowdin
from qstack.tools import correct_num_threads
from .utils import defaults


parser = argparse.ArgumentParser(description='This program computes the SPAHM(b) representation for a list of bonds')
parser.add_argument('--mol',           type=str,            dest='filename',       required=True,                    help='path to a list of molecular structures in xyz format and indices of bonds in question')
parser.add_argument('--guess',         type=str,            dest='guess',          default=defaults.guess,           help='initial guess')
parser.add_argument('--units',         type=str,            dest='units',          default='Angstrom',               help='the units of the input coordinates (default: Angstrom)')
parser.add_argument('--basis',         type=str,            dest='basis'  ,        default=defaults.basis,           help='AO basis set (default=MINAO)')
parser.add_argument('--charge',        type=str,            dest='charge',         default=None,                     help='file with a list of charges')
parser.add_argument('--spin',          type=str,            dest='spin',           default=None,                     help='file with a list of numbers of unpaired electrons')
parser.add_argument('--xc',            type=str,            dest='xc',             default=defaults.xc,              help=f'DFT functional for the SAD guess (default={defaults.xc})')
parser.add_argument('--dir',           type=str,            dest='dir',            default='./',                     help='directory to save the output in (default=current dir)')
parser.add_argument('--cutoff',        type=float,          dest='cutoff',         default=defaults.cutoff,          help=f'bond length cutoff in Å (default={defaults.cutoff})')
parser.add_argument('--bpath',         type=str,            dest='bpath',          default=defaults.bpath,           help=f'directory with basis sets (default={defaults.bpath})')
parser.add_argument('--same_basis',    action='store_true', dest='same_basis',     default=False,                    help='if to use generic CC.bas basis file for all atom pairs (Default: uses pair-specific basis, if exists)')
parser.add_argument('--omod',          type=str,            dest='omod',           default=defaults.omod, nargs='+', help=f'model for open-shell systems (alpha, beta, sum, diff, default={defaults.omod})')
parser.add_argument('--print',         type=int,            dest='print',          default=0,                        help='printing level')
parser.add_argument('--onlym0',        action='store_true', dest='only_m0',        default=False,                    help='use only functions with m=0')

args = parser.parse_args()
if args.print>0:
    print(vars(args))


def main():
    correct_num_threads()

    xyzlistfile = args.filename
    xyzlist = np.loadtxt(xyzlistfile, usecols=[0],   dtype=str, ndmin=1)
    bondidx = np.loadtxt(xyzlistfile, usecols=[1,2], dtype=int, ndmin=2)-1
    charge  = utils.get_chsp(args.charge, len(xyzlist))
    spin    = utils.get_chsp(args.spin,   len(xyzlist))
    if args.spin is None:
        args.omod = [None]

    mols = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units)
    dms  = utils.mols_guess(mols, xyzlist, args.guess,
                            xc=args.xc, spin=args.spin, printlevel=args.print)
    mybasis, idx, M = dmbb.read_basis_wrapper_pairs(mols, bondidx, args.bpath, args.only_m0, args.print, same_basis=args.same_basis)

    for j, (bondij, mol, dm, fname) in enumerate(zip(bondidx, mols, dms, xyzlist, strict=True)):
        if args.print>0:
            print('mol', j, flush=True)
        q = [mol.atom_symbol(i) for i in range(mol.natm)]
        r = mol.atom_coords(unit='ANG')
        vec = []
        for omod in args.omod:
            DM = utils.dm_open_mod(dm, omod) if args.spin is not None else dm
            L = lowdin.Lowdin_split(mol, DM)
            vec.append(dmbb.repr_for_bond(*bondij, L, mybasis, idx, q, r, args.cutoff)[0][0])
        vec = np.hstack(vec)
        outname = f'{args.dir}/{os.path.basename(fname)}_{bondij[0]+1}_{bondij[1]+1}'
        if args.spin:
            outname = outname+'_'+'_'.join(args.omod)
        if args.print>1:
            print(outname)
        np.save(outname, vec)


if __name__ == "__main__":
  main()
