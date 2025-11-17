"""Representation for a specific bond in a molecule."""

import os
import numpy as np
from . import utils, dmb_rep_bond as dmbb, lowdin
from qstack.tools import correct_num_threads
from .utils import defaults
from .parser import SpahmParser


def get_spahm_b_selected(mols, bondidx, xyzlist,
                         readdm=None, guess=defaults.guess, xc=defaults.xc, spin=None,
                         cutoff=defaults.cutoff, printlevel=0, omods=defaults.omod,
                         bpath=defaults.bpath, only_m0=False, same_basis=False):
    """Compute SPAHM(b) representations for specific bonds in molecules.

    Generates bond-centered representations for user-specified atom pairs across
    a dataset of molecules, useful for targeted bond analysis.

    Args:
        mols (list): List of pyscf Mole objects.
        bondidx (numpy ndarray): 2D array (nmols, 2) of 0-indexed atom pairs defining bonds.
        xyzlist (list): List of XYZ filenames corresponding to mols.
        readdm (str, optional): Directory to load pre-computed density matrices. Defaults to None.
        guess (str): Guess Hamiltonian method name. Defaults to defaults.guess.
        xc (str): Exchange-correlation functional. Defaults to defaults.xc.
        spin (numpy ndarray, optional): Array of numbers of unpaired electrons per molecule. Defaults to None.
        cutoff (float): Maximum bond distance in Å. Defaults to defaults.cutoff.
        printlevel (int): Verbosity level. Defaults to 0.
        omods (list): Open-shell modes (e.g. 'alpha', 'beta'). Defaults to defaults.omod.
        bpath (str): Path to bond basis set directory. Defaults to defaults.bpath.
        only_m0 (bool): Use only m=0 basis functions. Defaults to False.
        same_basis (bool): Use generic CC.bas for all pairs. Defaults to False.

    Returns:
        list: List of (filename, representation) tuples for each specified bond.
    """
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


def _get_arg_parser():
    """Parse CLI arguments."""
    parser = SpahmParser(description='This program computes the SPAHM(b) representation for a list of bonds', bond=True)
    parser.remove_argument('elements')
    parser.remove_argument('only_z')
    parser.remove_argument('name_out')
    parser.remove_argument('zeros')
    parser.remove_argument('pairfile')
    parser.remove_argument('dump_and_exit')
    parser.remove_argument('split')
    parser.remove_argument('merge')
    parser.remove_argument('with_symbols')
    parser.add_argument('--mol',     type=str,    dest='filename',  required=True,    help='path to a list of molecular structures in xyz format and indices of bonds in question')
    parser.add_argument('--charge',  type=str,    dest='charge',    default=None,     help='file with a list of charges')
    parser.add_argument('--spin',    type=str,    dest='spin',      default=None,     help='file with a list of numbers of unpaired electrons')
    parser.add_argument('--dir',     type=str,    dest='dir',       default='./',     help='directory to save the output in')
    return parser


def main():
    """Command-line interface for computing SPAHM(b) representations for specific bonds.

    Reads a file listing XYZ structures and bond indices, computes representations
    for each specified bond, and saves them to individual files. The input file format
    is: XYZ_path atom1_index atom2_index (1-indexed).

    Args:
        None: Parses command-line arguments.

    Output:
        Saves bond representations to numpy files in specified directory.
    """
    args = _get_arg_parser().parse_args()
    if args.print>0:
        print(vars(args))
    correct_num_threads()

    xyzlistfile = args.filename
    xyzlist = np.loadtxt(xyzlistfile, usecols=[0],   dtype=str, ndmin=1)
    bondidx = np.loadtxt(xyzlistfile, usecols=[1,2], dtype=int, ndmin=2)-1
    charge  = utils.get_chsp(args.charge, len(xyzlist))
    spin    = utils.get_chsp(args.spin,   len(xyzlist))

    mols = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units, ecp=args.ecp)

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
