#!/usr/bin/env python3

import argparse
import numpy as np
from qstack.tools import correct_num_threads
from . import utils, dmb_rep_bond as dmbb
from .utils import defaults

def bond(mols, dms,
         bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
         spin=None, elements=None, only_m0=False, zeros=False, split=False, printlevel=0):

    elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, bpath, only_m0, printlevel,
                                                                     elements=elements, cutoff=cutoff)
    if spin is None:
        omods = [None]
    qqs = qqs0 if zeros else qqs4q
    maxlen = max([dmbb.bonds_dict_init(qqs[q0], M)[1] for q0 in elements])
    natm   = max([mol.natm for mol in mols])
    allvec = np.zeros((len(omods), len(mols), natm, maxlen))

    for imol, (mol, dm) in enumerate(zip(mols,dms)):
        if printlevel>0: print('mol', imol, flush=True)
        for iomod, omod in enumerate(omods):
            DM  = utils.dm_open_mod(dm, omod) if spin else dm
            vec = dmbb.repr_for_mol(mol, DM, qqs, M, mybasis, idx, maxlen, cutoff)
            allvec[iomod,imol,:len(vec)] = vec

    if split is False:
        shape  = (len(omods), -1, maxlen)
        atidx  = np.where(np.array([[1]*mol.natm + [0]*(natm-mol.natm) for mol in mols]).flatten())
        allvec = allvec.reshape(shape)[:,atidx,:].reshape(shape)
    return allvec


def main():
    parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
    parser.add_argument('--mol',      type=str,            dest='filename',  required=True,                    help='file containing a list of molecular structures in xyz format')
    parser.add_argument('--guess',    type=str,            dest='guess',     default=defaults.guess,           help='initial guess type')
    parser.add_argument('--basis',    type=str,            dest='basis'  ,   default=defaults.basis,           help='AO basis set (default=MINAO)')
    parser.add_argument('--charge',   type=str,            dest='charge',    default=None,                     help='file with a list of charges')
    parser.add_argument('--spin',     type=str,            dest='spin',      default=None,                     help='file with a list of numbers of unpaired electrons')
    parser.add_argument('--xc',       type=str,            dest='xc',        default=defaults.xc,              help='DFT functional for the SAD guess (default=HF)')
    parser.add_argument('--dir',      type=str,            dest='dir',       default='./',                     help='directory to save the output in (default=current dir)')
    parser.add_argument('--cutoff',   type=float,          dest='cutoff',    default=defaults.cutoff,          help='bond length cutoff (A)')
    parser.add_argument('--bpath',    type=str,            dest='bpath',     default=defaults.bpath,           help='dir with basis sets')
    parser.add_argument('--omod',     type=str,            dest='omod',      default=defaults.omod, nargs='+', help='model for open-shell systems (alpha, beta, sum, diff')
    parser.add_argument('--print',    type=int,            dest='print',     default=0,                        help='printlevel')
    parser.add_argument('--zeros',    action='store_true', dest='zeros',     default=False,                    help='if use a version with more padding zeros')
    parser.add_argument('--split',    action='store_true', dest='split',     default=False,                    help='if split into molecules')
    parser.add_argument('--merge',    action='store_true', dest='merge',     default=False,                    help='if merge different omods')
    parser.add_argument('--onlym0',   action='store_true', dest='only_m0',   default=False,                    help='if use only fns with m=0')
    parser.add_argument('--savedm',   action='store_true', dest='savedm',    default=False,                    help='if save dms')
    parser.add_argument('--readdm',   type=str,            dest='readdm',    default=None,                     help='dir to read dms from')
    parser.add_argument('--elements', type=str,            dest='elements',  default=None,  nargs='+',         help="the elements contained in the database")
    parser.add_argument('--name',       dest='name_out',   required=True,                         type=str, help='name of the output files (for timing).')
    args = parser.parse_args()
    if args.print>0: print(vars(args))
    correct_num_threads()

    xyzlistfile = args.filename
    xyzlist = utils.get_xyzlist(xyzlistfile)
    charge  = utils.get_chsp(args.charge, len(xyzlist))
    spin    = utils.get_chsp(args.spin,   len(xyzlist))
    mols    = utils.load_mols(xyzlist, charge, spin, args.basis, args.print)
    dms     = utils.mols_guess(mols, xyzlist, args.guess,
                               xc=defaults.xc, spin=args.spin, readdm=args.readdm, printlevel=args.print)
    allvec  = bond(mols, dms, args.bpath, args.cutoff, args.omod,
                   spin=args.spin, elements=args.elements,
                   only_m0=args.only_m0, zeros=args.zeros, split=args.split, printlevel=args.print)

    if args.print>1: print(allvec.shape)
    
    

    if args.spin:
        if args.merge is False:
            for omod, vec in zip(args.omod, allvec):
                np.save(args.dir+'/mygreatrepresentation_'+omod, vec)
        else:
            np.save(args.dir+'/mygreatrepresentation_'+'_'.join(args.omod), allvec)
    else:
        np.save(args.dir+'/mygreatrepresentation', allvec[0])


if __name__ == "__main__":
    main()

