import os
import argparse
import numpy as np
from qstack.tools import correct_num_threads
from . import utils
from .utils import defaults
from .compute_rho_spahm import get_repr


def main(args=None):
    parser = argparse.ArgumentParser(description='This program computes the SPAHM(b) representation for a given molecular system or a list of thereof')
    parser.add_argument('--mol',           dest='filename',      type=str,            required=True,                    help='path to an xyz file / to a list of molecular structures in xyz format')
    parser.add_argument('--name',          dest='name_out',      type=str,            required=True,                    help='name of the output file')
    parser.add_argument('--guess',         dest='guess',         type=str,            default=defaults.guess,           help='initial guess')
    parser.add_argument('--units',         dest='units',         type=str,            default='Angstrom',               help='the units of the input coordinates (default: Angstrom)')
    parser.add_argument('--basis',         dest='basis'  ,       type=str,            default=defaults.basis,           help='AO basis set (default=MINAO)')
    parser.add_argument('--ecp',           dest='ecp',           type=str,            default=None,                     help='effective core potential to use (default: None)')
    parser.add_argument('--charge',        dest='charge',        type=str,            default="None",                   help='charge / path to a file with a list of thereof')
    parser.add_argument('--spin',          dest='spin',          type=str,            default="None",                   help='number of unpaired electrons / path to a file with a list of thereof')
    parser.add_argument('--xc',            dest='xc',            type=str,            default=defaults.xc,              help=f'DFT functional for the SAD guess (default={defaults.xc})')
    parser.add_argument('--dir',           dest='dir',           type=str,            default='./',                     help='directory to save the output in (default=current dir)')
    parser.add_argument('--cutoff',        dest='cutoff',        type=float,          default=defaults.cutoff,          help=f'bond length cutoff in Å (default={defaults.cutoff})')
    parser.add_argument('--bpath',         dest='bpath',         type=str,            default=defaults.bpath,           help=f'directory with basis sets (default={defaults.bpath})')
    parser.add_argument('--omod',          dest='omod',          type=str, nargs='+', default=defaults.omod,            help=f'model for open-shell systems (alpha, beta, sum, diff, default={defaults.omod})')
    parser.add_argument('--print',         dest='print',         type=int,            default=0,                        help='printing level')
    parser.add_argument('--zeros',         dest='zeros',         action='store_true', default=False,                    help='use a version with more padding zeros')
    parser.add_argument('--split',         dest='split',         action='count',      default=0,                        help='split into molecules (use twice to also split the output in one file per molecule)')
    parser.add_argument('--merge',         dest='merge',         action='store_true', default=True,                     help='merge different omods')
    parser.add_argument('--symbols',       dest='with_symbols',  action='store_true', default=False,                    help='if save tuples with (symbol, vec) for all atoms')
    parser.add_argument('--onlym0',        dest='only_m0',       action='store_true', default=False,                    help='use only functions with m=0')
    parser.add_argument('--savedm',        dest='savedm',        action='store_true', default=False,                    help='save density matrices')
    parser.add_argument('--readdm',        dest='readdm',        type=str,            default=None,                     help='directory to read density matrices from')
    parser.add_argument('--elements',      dest='elements',      type=str, nargs='+', default=None,                     help='the elements to limit the representation for')
    parser.add_argument('--pairfile',      dest='pairfile',      type=str,            default=None,                     help='path to the atom pair file')
    parser.add_argument('--dump_and_exit', dest='dump_and_exit', action='store_true', default=False,                    help='write the atom pair file and exit if --pairfile is set')
    parser.add_argument('--same_basis',    dest='same_basis',    action='store_true', default=False,                    help='if to use generic CC.bas basis file for all atom pairs (Default: uses pair-specific basis, if exists)')
    parser.add_argument('--only-z',        dest='only_z',        type=str, nargs='+', default=None,                     help="restrict the representation to one or several atom types")
    args = parser.parse_args(args=args)
    if args.print>0:
        print(vars(args))
    correct_num_threads()

    if args.name_out is None:
        args.name_out = os.path.splitext(args.filename)[0]

    if args.filename.endswith('xyz'):
        xyzlist = [args.filename]
        charge  = np.array([int(args.charge) if (args.charge != "None") else 0])
        spin    = np.array([int(args.spin)   if (args.spin != "None") else None])
    else:
        xyzlistfile = args.filename
        xyzlist = utils.get_xyzlist(xyzlistfile)
        if args.charge is not None:
            charge = utils.get_chsp(args.charge, len(xyzlist))
        else:
            charge = np.full(len(xyzlist), None, dtype=object)
        if args.spin is not None:
            spin = utils.get_chsp(args.spin, len(xyzlist))
        else:
            spin = np.full(len(xyzlist), None, dtype=object)

    mols = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units, ecp=args.ecp)

    reps = get_repr("bond",
        mols=mols, xyzlist=xyzlist, guess=args.guess, xc=args.xc, spin=spin,
        readdm=args.readdm, printlevel=args.print,
        pairfile=args.pairfile, dump_and_exit=args.dump_and_exit, same_basis=args.same_basis,
        bpath=args.bpath, cutoff=args.cutoff, omods=args.omod, with_symbols=args.with_symbols,
        elements=args.elements, only_m0=args.only_m0, zeros=args.zeros, split=(args.split>0), only_z=args.only_z,
    )

    if args.print > 0:
        print(reps.shape)
    if args.merge:
        if (spin == None).all():
            mod_iter = [(reps, '')]
        else:
            mod_iter = [(reps, '_'+'_'.join(args.omod))]
    else:
        mod_iter = [(modvec, '_'+omod) for modvec, omod in zip(reps, args.omod, strict=True)]

    for modvec, mod_suffix in mod_iter:
        if args.split >=2:
            for mol_i, molvec in enumerate(modvec):
                filename = xyzlist[mol_i]
                basename = os.path.splitext(os.path.basename(filename))[0]
                np.save(args.name_out + '_' + basename + mod_suffix, molvec)
        else:
            np.save(args.name_out + mod_suffix, modvec)

if __name__ == "__main__":
    main()
