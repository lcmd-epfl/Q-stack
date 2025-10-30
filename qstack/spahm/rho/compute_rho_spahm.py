#!/usr/bin/env python3

import os
import argparse
import numpy as np
from qstack.tools import correct_num_threads
from . import utils, dmb_rep_bond as dmbb
from . import dmb_rep_atom as dmba
from .utils import defaults

def spahm_a_b(rep_type, mols, dms,
         bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
         elements=None, only_m0=False, zeros=False, printlevel=0,
         auxbasis = 'ccpvdzjkfit', model='lowdin-long-x',
         pairfile=None, dump_and_exit=False, same_basis=False, only_z=[]):
    """ Computes SPAHM(a,b) representations for a set of molecules.

    Args:
        - rep_type (str) : the representation type ('atom' or 'bond' centered)
        - mols (list): the list of molecules (pyscf.Mole objects)
        - dms (list of numpy.ndarray): list of guess density matrices for each molecule
        - bpath (str): path to the directory containing bond-optimized basis-functions (.bas)
        - cutoff (float): the cutoff distance (angstrom) between atoms to be considered as bond
        - omods (list of str): the selected mode for open-shell computations
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
    maxlen = 0 # This needs fixing `UnboundLocalError`
    if rep_type == 'bond':
        elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, bpath, only_m0, printlevel,
                                                                         elements=elements, cutoff=cutoff,
                                                                         pairfile=pairfile, dump_and_exit=dump_and_exit, same_basis=same_basis)
        qqs = qqs0 if zeros else qqs4q
        maxlen = max([dmbb.bonds_dict_init(qqs[q0], M)[1] for q0 in elements])
    elif rep_type == 'atom':
        if elements is None:
            elements = set()
            for mol in mols:
                elements.update(mol.elements)
        elements = sorted(set(elements))
        df_wrapper, sym_wrapper = dmba.get_model(model)
        ao, ao_len, idx, M = dmba.get_basis_info(elements, auxbasis)
        maxlen = sum([len(v) for v in idx.values()])

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
        if len(only_z) >0:
            only_i = [i for i,z in enumerate(mol.elements) if z in only_z]
        else:
            only_i = range(mol.natm)

        for iomod, omod in enumerate(omods):
            DM  = utils.dm_open_mod(dm, omod)
            vec = None # This too !!! (maybe a wrapper or dict)
            if rep_type == 'bond':
                vec = dmbb.repr_for_mol(mol, DM, qqs, M, mybasis, idx, maxlen, cutoff, only_z=only_z)
            elif rep_type == 'atom':
                c_df = df_wrapper(mol, DM, auxbasis, only_i=only_i)
                vec = sym_wrapper(c_df, mol, idx, ao, ao_len, M, elements)
            allvec[iomod,imol,:len(vec)] = vec

    return allvec

def get_repr(rep_type, mols, xyzlist, guess,  xc=defaults.xc, spin=None, readdm=None,
             pairfile=None, dump_and_exit=False, same_basis=True,
             bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
             elements=None, only_m0=False, zeros=False, split=False, printlevel=0,
             auxbasis='ccpvdzjkfit', model="lowdin-long-x",
             with_symbols=False, only_z=[], merge=True):
    """ Computes and reshapes an array of SPAHM(a,b) representations

    Args:
        - rep_type (str) : the representation type ('atom' or 'bond' centered)
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
        A numpy.ndarray with all representations with shape (Nmods,Nmolecules,Natoms,Nfeatures)
        with:
          - Nmods: the alpha and beta components of the representation
          - Nmolecules: the number of molecules in the set
          - Natoms: the number of atoms in one molecule
          - Nfeatures: the number of features (for each omod)
        reshaped according to:
            - if split==False: collapses Nmolecules and returns a single np.ndarray (Nmods,Natoms,Nfeatures) (where Natoms is the total number of atoms in the set of molecules)
            - if merge==True: collapses the Nmods axis into the Nfeatures axis
            - if with_symbols==True: returns (for each molecule (Natoms, 2) containging the atom symbols along 1st dim and one of the above arrays
    """
    if not dump_and_exit:
        dms     = utils.mols_guess(mols, xyzlist, guess,
                               xc=defaults.xc, spin=spin, readdm=readdm, printlevel=printlevel)
    else:
        dms = []

    if len(only_z) > 0:
        all_atoms   = [ [sym for sym in mol.elements if sym in only_z] for mol in mols]
    else:
        all_atoms   = [mol.elements for mol in mols]

    spin = np.array(spin) ## a bit dirty but couldn't find a better way to ensure Iterable type!
    if (spin == None).all():
        omods = [None]

    allvec  = spahm_a_b(rep_type, mols, dms, bpath, cutoff, omods,
                   model=model,
                   elements=elements, only_m0=only_m0,
                   zeros=zeros, printlevel=printlevel,
                   auxbasis=auxbasis,
                   pairfile=pairfile, dump_and_exit=dump_and_exit, same_basis=same_basis, only_z=only_z)
    maxlen=allvec.shape[-1]
    natm = allvec.shape[-2]
    if split:
        # note: whenever there is zip() on a molecule, this automatically removes the padding
        if merge:
            allvec = np.concatenate(allvec, axis=2)
            if with_symbols:
                allvec = np.array([
                    np.array(list(zip(mol_atoms,molvec)), dtype=object)
                    for molvec,mol_atoms in zip(allvec, all_atoms)
                ], dtype=object)
            else:
                allvec = np.array([
                    molvec[:len(mol_atoms)]
                    for molvec,mol_atoms in zip(allvec, all_atoms)
                ], dtype=object)

        else:
            if with_symbols:
                allvec = np.array([
                    [
                        np.array(list(zip(mol_atoms,molvec)), dtype=object)
                        for molvec,mol_atoms in zip(modvec,all_atoms)
                    ]
                    for modvec in allvec
                ], dtype=object)
            else:
                allvec = np.array([
                    [
                        molvec[:len(mol_atoms)]
                        for molvec,mol_atoms in zip(modvec,all_atoms)
                    ]
                    for modvec in allvec
                ], dtype=object)

    else:
        natm_tot = sum(len(elems) for elems in all_atoms)
        allvec_new = np.empty_like(allvec, shape=(len(omods), natm_tot, maxlen))
        atm_i = 0
        for mol_i, elems in enumerate(all_atoms):
            allvec_new[:, atm_i:atm_i+len(elems), :] = allvec[:, mol_i, :len(elems), :]
            atm_i += len(elems)
        allvec = allvec_new; del allvec_new
        all_atoms = sum(all_atoms, start=[])

        if merge:
            allvec = np.hstack(allvec)
            if with_symbols:
                allvec = np.array(list(zip(all_atoms, allvec)), dtype=object)
        else:
            if with_symbols:
                allvec = np.array([
                    np.array(list(zip(all_atoms, modvec)), dtype=object)
                    for modvec in allvec
                ], dtype=object)

    return allvec

def main(args=None):
    parser = argparse.ArgumentParser(description='This program computes the SPAHM(a,b) representations for a given molecular system or a list thereof')
    parser.add_argument('--mol',           dest='filename',      type=str,            required=True,                    help='path to an xyz file / to a list of molecular structures in xyz format')
    parser.add_argument('--name',          dest='name_out',      type=str,            required=True,                    help='name of the output file')
    parser.add_argument('--rep',           dest='rep',           type=str, choices=['atom', 'bond'], required=True,     help='the type of representation')
    parser.add_argument('--guess',         dest='guess',         type=str,            default=defaults.guess,           help='initial guess')
    parser.add_argument('--units',         dest='units',         type=str,            default='Angstrom',               help='the units of the input coordinates (default: Angstrom)')
    parser.add_argument('--basis',         dest='basis'  ,       type=str,            default=defaults.basis,           help='AO basis set (default=MINAO)')
    parser.add_argument('--ecp',           dest='ecp',           type=str,            default=None,                     help=f'effective core potential to use (default: None)')
    parser.add_argument('--charge',        dest='charge',        type=str,            default="None",                     help='charge / path to a file with a list of thereof')
    parser.add_argument('--spin',          dest='spin',          type=str,            default="None",                     help='number of unpaired electrons / path to a file with a list of thereof')
    parser.add_argument('--aux-basis',     dest='auxbasis',      type=str,            default=defaults.auxbasis,        help=f"auxiliary basis set for density fitting (default: {defaults.auxbasis})")
    parser.add_argument('--xc',            dest='xc',            type=str,            default=defaults.xc,              help=f'DFT functional for the SAD guess (default={defaults.xc})')
    parser.add_argument('--dir',           dest='dir',           type=str,            default='./',                     help=f'directory to save the output in (default=current dir)')
    parser.add_argument('--cutoff',        dest='cutoff',        type=float,          default=defaults.cutoff,          help=f'bond length cutoff in Å (default={defaults.cutoff})')
    parser.add_argument('--bpath',         dest='bpath',         type=str,            default=defaults.bpath,           help=f'directory with basis sets (default={defaults.bpath})')
    parser.add_argument('--omod',          dest='omod',          type=str, nargs='+', default=defaults.omod,            help=f'model for open-shell systems (alpha, beta, sum, diff, default={defaults.omod})')
    parser.add_argument('--model',         dest='model',         type=str,            default=defaults.model,           help=f'model for the atomic density fitting (default={defaults.model})')
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
    parser.add_argument('--only-z',        dest='only_z',        type=str, nargs='+', default=[],                       help="restrict the representation to one or several atom types")
    args = parser.parse_args(args=args)
    if args.print>0: print(vars(args))
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

    reps = get_repr(args.rep,
        mols, xyzlist, args.guess, xc=args.xc, spin=spin,
        readdm=args.readdm, printlevel=args.print,
        auxbasis=args.auxbasis, model=args.model,
        pairfile=args.pairfile, dump_and_exit=args.dump_and_exit, same_basis=args.same_basis,
        bpath=args.bpath, cutoff=args.cutoff, omods=args.omod, with_symbols=args.with_symbols,
        elements=args.elements, only_m0=args.only_m0, zeros=args.zeros, split=(args.split>0), only_z=args.only_z,
    )
    if args.print > 0: print(reps.shape)
    if args.merge:
        if (spin == None).all():
            mod_iter = [(reps, '')]
        else:
            mod_iter = [(reps, '_'+'_'.join(args.omod))]
    else:
        mod_iter = [(modvec, '_'+omod) for modvec, omod in zip(reps, args.omod)]

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
