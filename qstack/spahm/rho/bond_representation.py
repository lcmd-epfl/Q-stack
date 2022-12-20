#!/usr/bin/env python3

import argparse
import numpy as np
from modules import utils, dmb_rep_bond as dmbb
from qstack.tools import correct_num_threads

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
parser.add_argument('--mol',        type=str,            dest='filename',  required=True,                        help='file containing a list of molecular structures in xyz format')
parser.add_argument('--charge',     type=str,            dest='charge',    default=None,                         help='file with a list of charges')
parser.add_argument('--spin',       type=str,            dest='spin',      default=None,                         help='file with a list of numbers of unpaired electrons')
parser.add_argument('--guess',      type=str,            dest='guess',     default='lb',                         help='initial guess type')
parser.add_argument('--basis',      type=str,            dest='basis'  ,   default='minao',                      help='AO basis set (default=MINAO)')
parser.add_argument('--func',       type=str,            dest='func',      default='hf',                         help='DFT functional for the SAD guess (default=HF)')
parser.add_argument('--dir',        type=str,            dest='dir',       default='./',                         help='directory to save the output in (default=current dir)')
parser.add_argument('--cutoff',     type=float,          dest='cutoff',    default=5.0,                          help='bond length cutoff')
parser.add_argument('--bpath',      type=str,            dest='bpath',     default='basis/optimized/',           help='dir with basis sets')
parser.add_argument('--omod',       type=str,            dest='omod',      default=['alpha','beta'], nargs='+',  help='model for open-shell systems (alpha, beta, sum, diff')
parser.add_argument('--print',      type=int,            dest='print',     default=0,                            help='printlevel')
parser.add_argument('--zeros',      action='store_true', dest='zeros',     default=False,                        help='if use a version with more padding zeros')
parser.add_argument('--split',      action='store_true', dest='split',     default=False,                        help='if split into molecules')
parser.add_argument('--onlym0',     action='store_true', dest='only_m0',   default=False,                        help='if use only fns with m=0')
parser.add_argument('--save',       action='store_true', dest='save',      default=False,                        help='if save dms')
parser.add_argument('--readdm',     type=str,            dest='readdm',    default=None,                         help='dir to read dms from')
parser.add_argument('--elements',   type=str,            dest='elements',  default=None,  nargs='+',             help="the elements contained in the database")


args = parser.parse_args()
if args.print>0: print(vars(args))


def main():
  correct_num_threads()

  xyzlistfile = args.filename
  xyzlist = utils.get_xyzlist(xyzlistfile)
  charge  = utils.get_chsp(args.charge, len(xyzlist))
  spin    = utils.get_chsp(args.spin,   len(xyzlist))

  mols, dms = utils.mols_guess(xyzlist, charge, spin, args)
  elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, args.bpath, args.only_m0, args.print, elements=args.elements, cutoff=args.cutoff)
  qqs = qqs0 if args.zeros else qqs4q

  maxlen = max([dmbb.bonds_dict_init(qqs[q0], M)[1] for q0 in elements ])

  for omod in args.omod:
    if args.split:
      natm   = max([mol.natm for mol in mols])
      allvec = np.zeros((len(mols), natm, maxlen))
    else:
      allvec = []

    for i,(mol,dm) in enumerate(zip(mols,dms)):
      if args.print>0: print('mol', i, flush=True)

      if args.spin: dm = utils.dm_open_mod(dm, omod)

      vec = dmbb.repr_for_mol(mol, dm, qqs, M, mybasis, idx, maxlen, args.cutoff)
      if args.split:
        allvec[i,:len(vec),:] = vec
      else:
        allvec.append(vec)

    if not args.split:
      allvec = np.vstack(allvec)


    if args.print>1: print(allvec.shape)
    if args.spin:
        np.save(args.dir+'/mygreatrepresentation_'+omod, allvec)
    else:
        np.save(args.dir+'/mygreatrepresentation', allvec)
        break


if __name__ == "__main__":
  main()

