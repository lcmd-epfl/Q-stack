#!/usr/bin/env python3

import argparse
import numpy as np
from . import utils, dmb_rep_bond as dmbb, lowdin
from qstack.tools import correct_num_threads
from .utils import defaults

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
parser.add_argument('--mol',        type=str,            dest='filename',  required=True,                        help='file containing a list of molecular structures in xyz format')
parser.add_argument('--charge',     type=str,            dest='charge',    default=None,                         help='file with a list of charges')
parser.add_argument('--spin',       type=str,            dest='spin',      default=None,                         help='file with a list of numbers of unpaired electrons')
parser.add_argument('--guess',      type=str,            dest='guess',     default='lb',                         help='initial guess type')
parser.add_argument('--basis',      type=str,            dest='basis'  ,   default='minao',                      help='AO basis set (default=MINAO)')
parser.add_argument('--func',       type=str,            dest='func',      default='hf',                         help='DFT functional for the SAD guess (default=HF)')
parser.add_argument('--dir',        type=str,            dest='dir',       default='./',                         help='directory to save the output in (default=current dir)')
parser.add_argument('--cutoff',     type=float,          dest='cutoff',    default=5.0,                          help='bond length cutoff')
parser.add_argument('--bpath',      type=str,            dest='bpath',     default=defaults.bpath,           help='dir with basis sets')
parser.add_argument('--omod',       type=str,            dest='omod',      default=['alpha','beta'], nargs='+',  help='model for open-shell systems (alpha, beta, sum, diff')
parser.add_argument('--print',      type=int,            dest='print',     default=0,                            help='printlevel')
parser.add_argument('--onlym0',     action='store_true', dest='only_m0',   default=False,                        help='if use only fns with m=0')
parser.add_argument('--elements',   type=str,            dest='elements',  default=None,  nargs='+',             help="the elements contained in the database")
args = parser.parse_args()
if args.print>0: print(vars(args))


def main():
    correct_num_threads()

    xyzlistfile = args.filename
    xyzlist = np.loadtxt(xyzlistfile, usecols=[0],   dtype=str, ndmin=1)
    bondidx = np.loadtxt(xyzlistfile, usecols=[1,2], dtype=int, ndmin=2)-1
    charge  = utils.get_chsp(args.charge, len(xyzlist))
    spin    = utils.get_chsp(args.spin,   len(xyzlist))
    if args.spin is None:
        args.omod = [None]

    mols    = utils.load_mols(xyzlist, charge, spin, args.basis, args.print)
    dms     = utils.mols_guess(mols, xyzlist, args.guess,
                               xc=defaults.xc, spin=args.spin, printlevel=args.print)
    elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, args.bpath, args.only_m0, args.print, elements=args.elements)

    for i,(bondij, mol, dm, fname) in enumerate(zip(bondidx, mols, dms, xyzlist)):
        if args.print>0: print('mol', i, flush=True)
        q = [mol.atom_symbol(i) for i in range(mol.natm)]
        r = mol.atom_coords(unit='ANG')
        vec = []
        for omod in args.omod:
            DM = utils.dm_open_mod(dm, omod) if args.spin else dm
            L = lowdin.Lowdin_split(mol, DM)
            vec.append(dmbb.repr_for_bond(*bondij, L, mybasis, idx, q, r, args.cutoff)[0][0])
        vec = np.hstack(vec)
        if args.spin:
            np.save(fname+'_'+'_'.join(args.omod), vec)
        else:
            np.save(fname, vec)


if __name__ == "__main__":
  main()
