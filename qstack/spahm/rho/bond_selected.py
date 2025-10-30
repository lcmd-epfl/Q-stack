import os
import argparse
import numpy as np
from . import utils, dmb_rep_bond as dmbb, lowdin
from qstack.tools import correct_num_threads
from .utils import defaults


def get_spahm_b_selected(mols, bondidx, xyzlist,
                         readdm=None, guess=defaults.guess, xc=defaults.xc, spin=None,
                         cutoff=defaults.cutoff, printlevel=0, omods=defaults.omod,
                         bpath=defaults.bpath, only_m0=False, same_basis=False):

    if spin is None or (spin == None).all():
        omods = [None]

    mybasis, idx, _M = dmbb.read_basis_wrapper_pairs(mols, bondidx, bpath, only_m0, printlevel, same_basis=same_basis)
    dms = utils.mols_guess(mols, xyzlist, readdm=readdm, guess=guess, xc=xc, spin=spin, printlevel=printlevel)

    vecs = []
    for j, (bondij, mol, dm, fname) in enumerate(zip(bondidx, mols, dms, xyzlist, strict=True)):
        if printlevel>0:
            print('mol', j, flush=True)
        q = [mol.atom_symbol(i) for i in range(mol.natm)]
        r = mol.atom_coords(unit='ANG')
        vec = []
        bondi, bondj = bondij
        for omod in omods:
            L = lowdin.Lowdin_split(mol, utils.dm_open_mod(dm, omod))
            vec.append(dmbb.repr_for_bond(bondi, bondj, L, mybasis, idx, q, r, cutoff)[0][0])
        outname = f'{os.path.basename(fname)}_{bondi+1}_{bondj+1}'
        if omods != [None]:
            outname = outname+'_'+'_'.join(omods)
        vecs.append((outname, np.hstack(vec)))
    return vecs


def main():
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
    parser.add_argument('--readdm',        type=str,            dest='readdm',         default=None,                     help='directory to read density matrices from')
    parser.add_argument('--onlym0',        action='store_true', dest='only_m0',        default=False,                    help='use only functions with m=0')

    args = parser.parse_args()
    if args.print>0:
        print(vars(args))
    correct_num_threads()

    xyzlistfile = args.filename
    xyzlist = np.loadtxt(xyzlistfile, usecols=[0],   dtype=str, ndmin=1)
    bondidx = np.loadtxt(xyzlistfile, usecols=[1,2], dtype=int, ndmin=2)-1
    charge  = utils.get_chsp(args.charge, len(xyzlist))
    spin    = utils.get_chsp(args.spin,   len(xyzlist))

    mols = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units)

    V = get_spahm_b_selected(mols, bondidx, xyzlist,
                             readdm=args.readdm, guess=args.guess, xc=args.xc, spin=spin,
                             cutoff=args.cutoff, printlevel=args.print, omods=args.omod,
                             bpath=args.bpath, only_m0=args.only_m0, same_basis=args.same_basis)

    for (outname, vec) in V:
        outname = f'{args.dir}/{outname}'
        if args.print>1:
            print(outname)
        np.save(outname, vec)


if __name__ == "__main__":
    main()
