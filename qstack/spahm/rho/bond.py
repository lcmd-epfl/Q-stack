#!/usr/bin/env python3

import os
import argparse
import numpy as np
from itertools import chain
from qstack.tools import correct_num_threads
from . import utils, dmb_rep_bond as dmbb
from .utils import defaults

def bond(mols, dms,
         bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
         spin=None, elements=None, only_m0=False, zeros=False, printlevel=0,
         pairfile=None, dump_and_exit=False, same_basis=False, only_z=[]):
    """ Computes SPAHM-b representations for a set of molecules.

    Args:
        - mols (list): the list of molecules (pyscf.Mole objects)
        - dms (list of numpy.ndarray): list of guess density matrices for each molecule
        - bpath (str): path to the directory containing bond-optimized basis-functions (.bas)
        - cutoff (float): the cutoff distance (angstrom) between atoms to be considered as bond
        - omods (list of str): the selected mode for open-shell computations
        - spin (list of int): list of spins for each molecule
        - elements (list of str): list of all elements present in the set of molecules
        - only_m0 (bool): use only basis functions with `m=0`
        - zeros (bool): add zeros features for non-existing bond pairs
        - printlevel (int): level of verbosity
        - pairfile (str): path to the pairfile (if None, atom pairs are detected automatically)
        - dump_and_exit (bool): to save pairfile for the set of molecules (without generating representaitons)
        - same_basis (bool): to use the same bond-optimized basis function for all atomic pairs (ZZ.bas == CC.bas for any Z)
        - only_z (list of str): restrict the atomic representations to atom types in this list

    Returns:
        A numpy.ndarray with the atomic spahm-b representations for each molecule (Nmods,Nmolecules,NatomMax,Nfeatures).
        with:   - Nmods: the alpha and beta components of the representation
                - Nmolecules: the number of molecules in the set
                - NatomMax: the maximum number of atoms in one molecule
                - Nfeatures: the number of features (for each omods)
    """
    elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, bpath, only_m0, printlevel,
                                                                     elements=elements, cutoff=cutoff,
                                                                     pairfile=pairfile, dump_and_exit=dump_and_exit, same_basis=same_basis)
    if np.array(spin==None, ndmin=1).all():
        omods = [None]
    qqs = qqs0 if zeros else qqs4q
    maxlen = max([dmbb.bonds_dict_init(qqs[q0], M)[1] for q0 in elements])
    if len(only_z) > 0:
        print(f"Selecting atom-types in {only_z}")
        zinmols = []
        for mol in mols:
            zinmol = [sum(z == np.array(mol.elements)) for z in only_z]
            zinmols.append(sum(zinmol))
        natm  = max(zinmols)
    else:
        natm   = max([mol.natm for mol in mols])
        zinmols = [mol.natm for mol in mols]
    allvec = np.zeros((len(omods), len(mols), natm, maxlen))

    for imol, (mol, dm) in enumerate(zip(mols,dms)):
        if printlevel>0: print('mol', imol, flush=True)
        for iomod, omod in enumerate(omods):
            DM  = utils.dm_open_mod(dm, omod)
            vec = dmbb.repr_for_mol(mol, DM, qqs, M, mybasis, idx, maxlen, cutoff, only_z=only_z)
            allvec[iomod,imol,:len(vec)] = vec

    return allvec

def get_repr(mols, xyzlist, guess,  xc=defaults.xc, spin=None, readdm=None,
             pairfile=None, dump_and_exit=False, same_basis=True,
             bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
             elements=None, only_m0=False, zeros=False, split=False, printlevel=0,
             with_symbols=False, only_z=[], merge=True):
    """ Computes and reshapes an array of SPAHM-b representation

    Args:
        - mols (list): the list of molecules (pyscf.Mole objects)
        - xyzlist (list of str): list with the paths to the xyz files
        - guess (str): the guess Hamiltonian
        - xc (str): the exchange-correlation functionals
        - dms (list of numpy.ndarray): list of guess density matrices for each molecule
        - readdm (str): path to the .npy file containins density matrices
        - bpath (str): path to the directory containing bond-optimized basis-functions (.bas)
        - cutoff (float): the cutoff distance (angstrom) between atoms to be considered as bond
        - omods (list of str): the selected mode for open-shell computations
        - spin (list of int): list of spins for each molecule
        - elements (list of str): list of all elements present in the set of molecules
        - only_m0 (bool): use only basis functions with `m=0`
        - zeros (bool): add zeros features for non-existing bond pairs
        - printlevel (int): level of verbosity
        - pairfile (str): path to the pairfile (if None, atom pairs are detected automatically)
        - dump_and_exit (bool): to save pairfile for the set of molecules (without generating representaitons)
        - same_basis (bool): to use the same bond-optimized basis function for all atomic pairs (ZZ.bas == CC.bas for any Z)
        - only_z (list of str): restrict the atomic representations to atom types in this list
        - split (bool): to split the final array into molecules
        - with_symbols (bool): to associate atomic symbol to representations in final array
        - merge (bool): to concatenate alpha and beta representations to a single feature vector

    Returns:
        A numpy.ndarray with all representations with shape (Nmods,Nmolecules,NatomsMax,Nfeatures)
        with:   - Nmods: the alpha and beta components of the representation
                - Nmolecules: the number of molecules in the set
                - NatomMax: the maximum number of atoms in one molecule
                - Nfeatures: the number of features (for each omods)
        reshaped according to:
            - if split==True: collapses Nmolecules and returns (Nmods,Natoms,Nfeatures) (where Natoms is the total number of atoms in the set of molecules)
            - if merge==True: collapses Nmods and returns (Natoms,Nfeatures*2)
            - if with_symbols==True: returns (Natoms, 2) containging the atom symbols along 1st dim and one of the above arrays
    """
    if not dump_and_exit:
        dms     = utils.mols_guess(mols, xyzlist, guess,
                               xc=defaults.xc, spin=spin, readdm=readdm, printlevel=printlevel)
    else:
        dms = []

    if len(only_z) > 0:
        all_atoms   = np.array([z for mol in mols for z in mol.elements if z in only_z])
    else:
        all_atoms   = np.array([mol.elements for mol in mols])

    allvec  = bond(mols, dms, bpath, cutoff, omods,
                   spin=spin, elements=elements,
                   only_m0=only_m0, zeros=zeros, printlevel=printlevel,
                   pairfile=pairfile, dump_and_exit=dump_and_exit, same_basis=same_basis, only_z=only_z)
    maxlen=allvec.shape[-1]
    natm = allvec.shape[-2]

    if split is False:
        shape  = (len(omods), -1, maxlen)
        atidx  = np.where(np.array([[1]*len(zin) + [0]*(natm-len(zin)) for zin in all_atoms]).flatten())
        allvec = allvec.reshape(shape)[:,atidx,:].reshape(shape)
        all_atoms = list(chain.from_iterable(all_atoms))
        allvec = allvec.squeeze()
    elif with_symbols:
        msg = f"You can not use 'split=True' and 'with_symbols=True' at the same time!"
        raise RuntimeError()

    if printlevel>0: print(allvec.shape)

    if merge is True:
        allvec = np.hstack(allvec)
        if with_symbols:
            allvec = np.array([(z, v) for v,z in zip(allvec, all_atoms)], dtype=object)
    elif with_symbols:
        allvec = np.array([[(z, v) for v,z in zip(modvec, all_atoms)] for modvec in allvec], dtype=object)
    return allvec

def main():
    parser = argparse.ArgumentParser(description='This program computes the SPAHM(b) representation for a given molecular system or a list of thereof')
    parser.add_argument('--mol',           type=str,            dest='filename',       required=True,                    help='path to an xyz file / to a list of molecular structures in xyz format')
    parser.add_argument('--name',          type=str,            dest='name_out',       required=True,                    help='name of the output file')
    parser.add_argument('--guess',         type=str,            dest='guess',          default=defaults.guess,           help='initial guess')
    parser.add_argument('--units',         type=str,            dest='units',          default='Angstrom',               help='the units of the input coordinates (default: Angstrom)')
    parser.add_argument('--basis',         type=str,            dest='basis'  ,        default=defaults.basis,           help='AO basis set (default=MINAO)')
    parser.add_argument('--ecp',        dest='ecp',        default=None,                  type=str, help=f'effective core potential to use (default: None)')
    parser.add_argument('--charge',        type=str,            dest='charge',         default=None,                     help='charge / path to a file with a list of thereof')
    parser.add_argument('--spin',          type=str,            dest='spin',           default=None,                     help='number of unpaired electrons / path to a file with a list of thereof')
    parser.add_argument('--xc',            type=str,            dest='xc',             default=defaults.xc,              help=f'DFT functional for the SAD guess (default={defaults.xc})')
    parser.add_argument('--dir',           type=str,            dest='dir',            default='./',                     help=f'directory to save the output in (default=current dir)')
    parser.add_argument('--cutoff',        type=float,          dest='cutoff',         default=defaults.cutoff,          help=f'bond length cutoff in Å (default={defaults.cutoff})')
    parser.add_argument('--bpath',         type=str,            dest='bpath',          default=defaults.bpath,           help=f'directory with basis sets (default={defaults.bpath})')
    parser.add_argument('--omod',          type=str,            dest='omod',           default=defaults.omod, nargs='+', help=f'model for open-shell systems (alpha, beta, sum, diff, default={defaults.omod})')
    parser.add_argument('--print',         type=int,            dest='print',          default=0,                        help='printing level')
    parser.add_argument('--zeros',         action='store_true', dest='zeros',          default=False,                    help='use a version with more padding zeros')
    parser.add_argument('--split',         action='store_true', dest='split',          default=False,                    help='split into molecules')
    parser.add_argument('--merge',         action='store_true', dest='merge',          default=True,                     help='merge different omods')
    parser.add_argument('--symbols',       action='store_true', dest='with_symbols',   default=False,                    help='if save tuples with (symbol, vec) for all atoms')
    parser.add_argument('--onlym0',        action='store_true', dest='only_m0',        default=False,                    help='use only functions with m=0')
    parser.add_argument('--savedm',        action='store_true', dest='savedm',         default=False,                    help='save density matrices')
    parser.add_argument('--readdm',        type=str,            dest='readdm',         default=None,                     help='directory to read density matrices from')
    parser.add_argument('--elements',      type=str,            dest='elements',       default=None,  nargs='+',         help='the elements to limit the representation for')
    parser.add_argument('--pairfile',      type=str,            dest='pairfile',       default=None,                     help='path to the atom pair file')
    parser.add_argument('--dump_and_exit', action='store_true', dest='dump_and_exit',  default=False,                    help='write the atom pair file and exit if --pairfile is set')
    parser.add_argument('--same_basis',    action='store_true', dest='same_basis',     default=False,                    help='if to use generic CC.bas basis file for all atom pairs (Default: uses pair-specific basis, if exists)')
    parser.add_argument('--only-z',        type=str,            dest='only_z',         default=[],  nargs='+',           help="restrict the representation to one or several atom types")
    args = parser.parse_args()
    if args.print>0: print(vars(args))
    correct_num_threads()

    if args.name_out is None:
        args.name_out = os.path.splitext(args.filename)[0]

    if args.filename.endswith('xyz'):
        xyzlist = [args.filename]
        charge  = [int(args.charge) if args.charge is not None else 0]
        spin    = [int(args.spin)   if args.spin   is not None else None]
    else:
        xyzlistfile = args.filename
        xyzlist = utils.get_xyzlist(xyzlistfile)
        charge  = utils.get_chsp(args.charge, len(xyzlist))
        spin    = utils.get_chsp(args.spin,   len(xyzlist))
    mols    = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units, ecp=args.ecp)
    
    reps = get_repr(mols, xyzlist, args.guess, xc=args.xc, spin=spin, readdm=args.readdm, printlevel=args.print,
                      pairfile=args.pairfile, dump_and_exit=args.dump_and_exit, same_basis=args.same_basis,
                      bpath=args.bpath, cutoff=args.cutoff, omods=args.omod, with_symbols=args.with_symbols,
                      elements=args.elements, only_m0=args.only_m0, zeros=args.zeros, split=args.split)
    if args.print > 0: print(reps.shape)
    if args.merge:
        np.save(args.name_out+'_'+'_'.join(args.omod), reps)
    else:
        for vec, omod in zip(reps, args.omod):
            np.save(args.name_out+'_'+omod, vec)

if __name__ == "__main__":
    main()
