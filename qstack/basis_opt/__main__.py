#!/usr/bin/env python3

"""
The command-line launcher for the basis set optimisation function of qstack.basis_opt.opt

(can be called as `python3 -m qstack.basis_opt`, and has a cleaner import chain than `python3 -m qstack.basis_opt.opt`)
"""

import sys, argparse
from . import basis_tools as qbbt
from .opt import optimize_basis


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Optimize a density fitting basis set.')
    parser.add_argument('--elements',  type=str,   dest='elements',  nargs='+',    help='elements for optimization')
    parser.add_argument('--basis',     type=str,   dest='basis',     nargs='+',    help='initial df bases', required=True)
    parser.add_argument('--molecules', type=str,   dest='molecules', nargs='+',    help='molecules', required=True)
    parser.add_argument('--gtol',      type=float, dest='gtol',      default=1e-7, help='tolerance')
    parser.add_argument('--method',    type=str,   dest='method',    default='CG', help='minimization algoritm')
    parser.add_argument('--print',     type=int,   dest='print',     default=2,    help='printing level')
    parser.add_argument('--check', action='store_true', dest='check', default=False, help='check the gradient and exit')
    parser.add_argument('--output',    type=str,   dest='output',    default=None, help='optional file to write the optimised basis set to (nwchem format, which pyscf can read)')
    return parser


def main():
    """Run basis set optimization via command-line interface."""
    args = _get_arg_parser().parse_args()

    result = optimize_basis(args.elements, args.basis, args.molecules, args.gtol, args.method, check=args.check, printlvl=args.print)
    if args.check is False:
        if args.print==0:
            qbbt.printbasis(result, sys.stdout)
        if args.output is not None:
           with open(args.output, 'w') as f:
               qbbt.basis_as_nwchem(f, result, '[placeholder name for a qstack-optimised basis set]')
    else:
        gr_an, gr_num, gr_diff = result['an'], result['num'], result['diff']
        print('analytical gradient')
        print(gr_an)
        print('numerical gradient')
        print(gr_num)
        print('difference')
        print(gr_diff)
        print('relative difference')
        print(gr_diff/gr_num)
        print(flush=True)


if __name__ == "__main__":
    main()

