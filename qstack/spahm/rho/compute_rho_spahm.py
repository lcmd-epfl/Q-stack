"""Main computation routines for SPAHM(a,b) representations."""

import os
import itertools
import numpy as np
from qstack.tools import correct_num_threads
from . import utils, dmb_rep_bond as dmbb
from . import dmb_rep_atom as dmba
from .utils import defaults
from .parser import SpahmParser


def spahm_a_b(rep_type, mols, dms,
         bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
         elements=None, only_m0=False, zeros=False, printlevel=0,
         auxbasis=defaults.auxbasis, model=defaults.model,
         pairfile=None, dump_and_exit=False, same_basis=False, only_z=None):
    """Computes SPAHM(a) or SPAHM(b) representations for a set of molecules.

    Reference:
        K. R. Briling, Y. Calvino Alonso, A. Fabrizio, C. Corminboeuf,
        "SPAHM(a,b): Encoding the density information from guess Hamiltonian in quantum machine learning representations",
        J. Chem. Theory Comput. 20 1108–1117 (2024), doi:10.1021/acs.jctc.3c01040

    Args:
        rep_type (str): Representation type: 'atom' for SPAHM(a) or 'bond' for SPAHM(b).
        mols (list): List of pyscf Mole objects.
        dms (list): List of density matrices (2D or 3D numpy arrays) for each molecule.
        bpath (str): Directory path containing bond-optimized basis files (.bas) for SPAHM(b). Defaults to defaults.bpath.
        cutoff (float): Bond cutoff distance in Å for SPAHM(b). Defaults to defaults.cutoff.
        omods (list): Open-shell modes ('alpha', 'beta', 'sum', 'diff'). Defaults to defaults.omod.
        elements (list, optional): Element symbols present in dataset. Auto-detected if None. Defaults to None.
        only_m0 (bool): Use only m=0 angular momentum component for SPAHM(b). Defaults to False.
        zeros (bool): Pad with zeros for non-existent bond pairs in SPAHM(b). Defaults to False.
        printlevel (int): Verbosity level (0=silent, >0=verbose). Defaults to 0.
        auxbasis (str): Auxiliary basis set for SPAHM(a). Defaults to defaults.auxbasis.
        model (str): Atomic density fitting model for SPAHM(a). Defaults to defaults.model.
        pairfile (str, optional): Path to atom pair file for SPAHM(b). Atom pairs are computed from mols if None. Defaults to None.
        dump_and_exit (bool): Save atom pair file for SPAHM(b) to pairfile and exit without computing. Defaults to False.
        same_basis (bool): Use generic CC.bas for all atom pairs for SPAHM(b). Defaults to False.
        only_z (list, optional): Restrict to specific atom types. Defaults to None.

    Returns:
        numpy ndarray: 4D array (n_omods, n_mols, max_atoms, n_features) where:
        - n_omods: Number of open-shell components (1 for closed-shell, len(omods) for open-shell)
        - n_mols: Number of molecules in dataset
        - max_atoms: Maximum number of atoms/bonds across all molecules
        - n_features: Representation dimension
    """
    maxlen = 0
    if only_z is None:
        only_z = []
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

    for imol, (mol, dm) in enumerate(zip(mols, dms, strict=True)):
        if printlevel>0:
            print('mol', imol, flush=True)
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
             auxbasis=defaults.auxbasis, model=defaults.model,
             with_symbols=False, only_z=None, merge=True):
    """Computes and reshapes SPAHM(a) or SPAHM(b) representations with flexible output formats.

    High-level interface that handles density matrix computation, representation generation,
    and output formatting including splitting, symbol labeling, and merging options.

    Args:
        rep_type (str): Representation type ('atom' or 'bond').
        mols (list): List of pyscf Mole objects.
        xyzlist (list): List of XYZ file paths corresponding to mols.
        guess (str): Guess Hamiltonian name.
        xc (str): Exchange-correlation functional. Defaults to defaults.xc.
        spin (list, optional): List of spin multiplicities per molecule. Defaults to None.
        readdm (str, optional): Directory path to load pre-computed density matrices. Defaults to None.
        pairfile (str, optional): Path to atom pair file for SPAHM(b). Defaults to None.
        dump_and_exit (bool): Save atom pair file for SPAHM(b) to pairfile and exit without computing. Defaults to False.
        same_basis (bool): Use generic CC.bas for all atom pairs for SPAHM(b). Defaults to False.
        bpath (str): Directory path containing bond-optimized basis files (.bas) for SPAHM(b). Defaults to defaults.bpath.
        cutoff (float): Bond cutoff distance in Å for SPAHM(b). Defaults to defaults.cutoff.
        omods (list): Open-shell modes ('alpha', 'beta', 'sum', 'diff'). Defaults to defaults.omod.
        elements (list, optional): Element symbols in dataset. Auto-detected if None. Defaults to None.
        only_m0 (bool): Use only m=0 angular momentum component for SPAHM(b). Defaults to False.
        zeros (bool): Pad with zeros for non-existent bond pairs in SPAHM(b). Defaults to False.
        split (bool): Split output by molecule. Defaults to False.
        printlevel (int): Verbosity level. Defaults to 0.
        auxbasis (str): Auxiliary basis for SPAHM(a). Defaults to defaults.auxbasis.
        model (str): Atomic density fitting model for SPAHM(a). Defaults to defaults.model.
        with_symbols (bool): Include atomic symbols with representations. Defaults to False.
        only_z (list, optional): Restrict to specific atom types. Defaults to None.
        merge (bool): Merge alpha/beta into single vector. Defaults to True.

    Returns:
        numpy ndarray: Representation array with shape depending on options:
        - Base: (n_omods, n_mols, max_atoms, n_features)
        - If split=False: (n_omods, total_atoms, n_features) - all molecules concatenated
        - If merge=True: Features concatenated, omods dimension removed
        - If with_symbols=True: Object array with (symbol, vector) tuples per atom
        - If split=True and with_symbols=True: List format per molecule
    """
    if not dump_and_exit:
        dms = utils.mols_guess(mols, xyzlist, guess, xc=xc, spin=spin, readdm=readdm, printlevel=printlevel)
    else:
        dms = []

    if only_z is not None and len(only_z) > 0:
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
    if split:
        # note: whenever there is zip() on a molecule, this automatically removes the padding
        if merge:
            allvec = np.concatenate(allvec, axis=2)
            if with_symbols:
                allvec = np.array([
                    np.array(list(zip(mol_atoms, molvec, strict=False)), dtype=object)
                    for molvec,mol_atoms in zip(allvec, all_atoms, strict=True)
                ], dtype=object)
            else:
                allvec = np.array([
                    molvec[:len(mol_atoms)]
                    for molvec,mol_atoms in zip(allvec, all_atoms, strict=True)
                ], dtype=object)

        else:
            if with_symbols:
                allvec = np.array([
                    [
                        np.array(list(zip(mol_atoms, molvec, strict=False)), dtype=object)
                        for molvec,mol_atoms in zip(modvec, all_atoms, strict=True)
                    ]
                    for modvec in allvec
                ], dtype=object)
            else:
                allvec = np.array([
                    [
                        molvec[:len(mol_atoms)]
                        for molvec,mol_atoms in zip(modvec, all_atoms, strict=True)
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
        allvec = allvec_new
        del allvec_new
        all_atoms = list(itertools.chain.from_iterable(all_atoms))

        if merge:
            allvec = np.hstack(allvec)
            if with_symbols:
                allvec = np.array(list(zip(all_atoms, allvec, strict=True)), dtype=object)
        else:
            if with_symbols:
                allvec = np.array([
                    np.array(list(zip(all_atoms, modvec, strict=True)), dtype=object)
                    for modvec in allvec
                ], dtype=object)

    return allvec


def main(args=None):
    """Command-line interface for computing SPAHM representations (atom or bond centered).

    Unified CLI that supports both SPAHM(a) and SPAHM(b) computations with extensive
    options for molecular datasets, splitting, and output formatting.

    Args:
        args (list, optional): Command-line arguments. If None, uses sys.argv. Defaults to None.

    Returns:
        None: Saves representations to numpy files based on --name argument and options.
    """
    parser = SpahmParser(description='This program computes the SPAHM(a,b) representations for a given molecular system or a list thereof', unified=True, atom=True, bond=True)
    parser.add_argument('--rep',  dest='rep',  type=str, choices=['atom', 'bond'], required=True, help='the type of representation')
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

    reps = get_repr(args.rep,
        mols, xyzlist, args.guess, xc=args.xc, spin=spin,
        readdm=args.readdm, printlevel=args.print,
        auxbasis=args.auxbasis, model=args.model,
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
