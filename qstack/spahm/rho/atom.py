import argparse
import os
import numpy as np
from qstack import compound
from .utils import defaults
from .compute_rho_spahm import get_repr


def main(args=None):
    parser = argparse.ArgumentParser(description='This program computes the SPAHM(a) representation for a given molecular system')
    parser.add_argument('--mol',       dest='mol',           required=True,                        type=str, help="the path to the xyz file with the molecular structure")
    parser.add_argument('--guess',     dest='guess',         default=defaults.guess,               type=str, help=f"the initial guess Hamiltonian to be used (default: {defaults.guess})")
    parser.add_argument('--units',     dest='units',         default='Angstrom',                   type=str, help="the units of the input coordinates (default: Angstrom)")
    parser.add_argument('--basis-set', dest='basis',         default=defaults.basis,               type=str, help=f"basis set for computing density matrix (default: {defaults.basis})")
    parser.add_argument('--aux-basis', dest='auxbasis',      default=defaults.auxbasis,            type=str, help=f"auxiliary basis set for density fitting (default: {defaults.auxbasis})")
    parser.add_argument('--model',     dest='model',         default=defaults.model,               type=str, help=f"the model to use when creating the representation (default: {defaults.model})")
    parser.add_argument('--dm',        dest='dm',            default=None,                         type=str, help="a density matrix to load instead of computing the guess")
    parser.add_argument('--species',   dest='elements',      default=None, nargs='+',              type=str, help="the elements contained in the database")
    parser.add_argument('--only',      dest='only_z',        default=None, nargs='+',              type=str, help="The restricted list of elements for which you want to generate the representation")
    parser.add_argument('--charge',    dest='charge',        default=0,                            type=int, help='total charge of the system (default: 0)')
    parser.add_argument('--spin',      dest='spin',          default=None,                         type=int, help='number of unpaired electrons (default: None) (use 0 to treat a closed-shell system in a UHF manner)')
    parser.add_argument('--xc',        dest='xc',            default=defaults.xc,                  type=str, help=f'DFT functional for the SAD guess (default: {defaults.xc})')
    parser.add_argument('--ecp',       dest='ecp',           default=None,                         type=str, help='effective core potential to use (default: None)')
    parser.add_argument('--nameout',   dest='NameOut',       default=None,                         type=str, help='name of the output representations file.')
    parser.add_argument('--omod',      dest='omod',          default=defaults.omod,     nargs='+', type=str, help=f'model(s) for open-shell systems (alpha, beta, sum, diff, default: {defaults.omod})')
    parser.add_argument('--split',     dest='split',         action='count',  default=0,                     help='split into molecules (use twice to also split the output in one file per molecule)')
    parser.add_argument('--merge',     dest='merge',         action='store_true',                            help='merge different omods')
    parser.add_argument('--symbols',   dest='with_symbols',  action='store_true',                            help='if save tuples with (symbol, vec) for all atoms')
    parser.add_argument('--print',     dest='print',         default=0,                            type=int, help='printing level')
    args = parser.parse_args(args=args)
    if args.print>0:
        print(vars(args))

    mol = compound.xyz_to_mol(args.mol, args.basis, charge=args.charge, spin=args.spin, unit=args.units, ecp=args.ecp)

    if args.elements is None:
        elements = sorted(mol.elements)
    else:
        elements = args.elements

    representations = get_repr(rep_type="atom", mols=[mol], xyzlist=[args.mol], guess=args.guess,
                               xc=args.xc, spin=args.spin, readdm=args.dm,
                               omods=args.omod, elements=elements,
                               auxbasis=args.auxbasis, model=args.model,
                               only_z=args.only_z,
                               split=args.split, with_symbols=args.with_symbols, merge=args.merge,
                               printlevel=args.print)
    cwd = os.getcwd()
    mol_name = args.mol.split('/')[-1].split('.')[0]
    if args.NameOut is not None:
        name_out = args.NameOut
    else:
        name_out = 'X_'+mol_name
    if args.spin is not None:
        name_out = name_out+'_'+'_'.join(args.omod)
    path_out = os.path.join(cwd, name_out)
    np.save(path_out, representations)


if __name__ == '__main__':
    main()
