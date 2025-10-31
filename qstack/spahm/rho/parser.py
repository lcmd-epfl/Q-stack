import argparse
from qstack.tools import FlexParser
from .utils import defaults, omod_fns_dict
from .dmb_rep_atom import models_dict
from ..guesses import guesses_dict


class SpahmParser(FlexParser):
    """Custom argument parser for SPAHM command-line tools.

    Provides pre-configured argument sets for atomic and bond SPAHM computations
    with consistent interface across different entry points.

    Args:
        unified (bool): Enable unified file/list interface. Defaults to False.
        atom (bool): Add atom-specific arguments (auxbasis, model). Defaults to False.
        bond (bool): Add bond-specific arguments (cutoff, bpath, etc.). Defaults to False.
        **kwargs: Additional arguments passed to ArgumentParser.
    """
    def __init__(self, unified=False, atom=False, bond=False, **kwargs):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
        parser = self
        if unified:
            parser.add_argument('--mol',           dest='filename',      required=True,                            type=str,     help='path to an xyz file / to a list of molecular structures in xyz format')
            parser.add_argument('--charge',        dest='charge',        default="None",                           type=str,     help='charge / path to a file with a list of thereof')
            parser.add_argument('--spin',          dest='spin',          default="None",                           type=str,     help='number of unpaired electrons / path to a file with a list of thereof')
        parser.add_argument('--guess',     dest='guess', default=defaults.guess, choices=guesses_dict.keys(),      type=str,     help="the initial guess Hamiltonian to be used")
        parser.add_argument('--basis',             dest='basis',         default=defaults.basis,                   type=str,     help="basis set for computing density matrix")
        parser.add_argument('--xc',                dest='xc',            default=defaults.xc,                      type=str,     help='DFT functional for the SAD guess')
        parser.add_argument('--ecp',               dest='ecp',           default=None,                             type=str,     help='effective core potential to use')
        parser.add_argument('--readdm',            dest='readdm',        default=None,                             type=str,     help='directory to read density matrices from')
        parser.add_argument('--units', dest='units', default='Angstrom', choices=('Angstrom', 'Bohr'),             type=str,     help="the units of the input coordinates")
        parser.add_argument('--elements',          dest='elements',      default=None, nargs='+',                  type=str,     help="the elements contained in the database")
        parser.add_argument('--only-z',            dest='only_z',        default=None, nargs='+',                  type=str,     help="restrict the representation to one or several atom types")
        parser.add_argument('--omod', dest='omod', default=defaults.omod, choices=omod_fns_dict.keys(), nargs='+', type=str,     help='model(s) for open-shell systems (alpha, beta, sum, diff')
        parser.add_argument('--name',              dest='name_out',      default=None,                             type=str,     help='name of the output representations file.')
        parser.add_argument('--split',             dest='split',         default=0,      action='count',                         help='split into molecules (use twice to also split the output in one file per molecule)')
        parser.add_argument('--nomerge',           dest='merge',         action='store_false',                                   help='merge different omods')
        parser.add_argument('--symbols',           dest='with_symbols',  action='store_true',                                    help='if save tuples with (symbol, vec) for all atoms')
        parser.add_argument('--print',             dest='print',         default=0,                                type=int,     help='printing level')
        if atom:
            parser.add_argument('--aux-basis',     dest='auxbasis',      default=defaults.auxbasis,                type=str,     help="auxiliary basis set for density fitting")
            parser.add_argument('--model',       dest='model', default=defaults.model, choices=models_dict.keys(), type=str,     help='model for the atomic density fitting')
        if bond:
            parser.add_argument('--cutoff',        dest='cutoff',        default=defaults.cutoff,                  type=float,   help='bond length cutoff in Å')
            parser.add_argument('--bpath',         dest='bpath',         default=defaults.bpath,                   type=str,     help='directory with basis sets')
            parser.add_argument('--zeros',         dest='zeros',         action='store_true',                                    help='use a version with more padding zeros')
            parser.add_argument('--onlym0',        dest='only_m0',       action='store_true',                                    help='use only functions with m=0')
            parser.add_argument('--pairfile',      dest='pairfile',      default=None,                             type=str,     help='path to the atom pair file')
            parser.add_argument('--dump_and_exit', dest='dump_and_exit', action='store_true',                                    help='write the atom pair file and exit if --pairfile is set')
            parser.add_argument('--same_basis',    dest='same_basis',    action='store_true',                                    help='if to use generic CC.bas basis file for all atom pairs (Default: uses pair-specific basis, if exists)')
