#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
from qstack import compound, spahm
from . import utils, dmb_rep_atom as dmba
from .utils import defaults


def check_file(mol_file):
    if not os.path.isfile(mol_file):
        print(f"Error xyz-file {mol_file} not found !\nExiting !!", file=sys.stderr)
        exit(1)
    return mol_file


def get_repr(mol, elements, charge, spin,
             open_mod=defaults.omod, dm=None,
             guess=defaults.guess, model=defaults.model, xc=defaults.xc,
             auxbasis=defaults.auxbasis):

    # User-defined options
    elements = sorted(list(set(elements)))
    guess = spahm.guesses.get_guess(guess)
    model = dmba.get_model(model)
    df_wrapper, sym_wrapper = model
    ao, ao_len, idx, M = dmba.get_basis_info(elements, auxbasis)

    # Compute density matrices
    if dm is None:
        dm = spahm.compute_spahm.get_guess_dm(mol, guess, openshell=spin, xc=xc)

    rep = []
    for omod in open_mod:
        DM      = utils.dm_open_mod(dm, omod) if spin is not None else dm
        c_df    = df_wrapper(mol, DM, auxbasis)
        vectors = sym_wrapper(c_df, mol, idx, ao, ao_len, M, elements)
        if spin is None:
            rep = vectors
            break
        rep.append(vectors)
    rep = np.array(rep)
    if spin is not None:
        rep = np.hstack(rep)

    mrep = []
    for q, v in zip(mol.elements, rep):
        mrep.append(np.array((q, v)))
    return np.array(mrep)


def main():
    parser = argparse.ArgumentParser(description='This program computes the SPAHM(a) representation for a given molecular system')
    parser.add_argument('--mol',       dest='mol',       required=True,                        type=str, help="the path to the xyz file with the molecular structure")
    parser.add_argument('--guess',     dest='guess',     default=defaults.guess,               type=str, help=f"the initial guess Hamiltonian to be used (default: {defaults.guess})")
    parser.add_argument('--units',     dest='units',     default='Angstrom',                   type=str, help=f"the units of the input coordinates (default: Angstrom)")
    parser.add_argument('--basis-set', dest='basis',     default=defaults.basis,               type=str, help=f"basis set for computing density matrix (default: {defaults.basis})")
    parser.add_argument('--aux-basis', dest='auxbasis',  default=defaults.auxbasis,            type=str, help=f"auxiliary basis set for density fitting (default: {defaults.auxbasis})")
    parser.add_argument('--model',     dest='model',     default=defaults.model,               type=str, help=f"the model to use when creating the representation (default: {defaults.model})")
    parser.add_argument('--dm',        dest='dm',        default=None,                         type=str, help="a density matrix to load instead of computing the guess")
    parser.add_argument('--species',   dest='elements',  default=defaults.elements, nargs='+', type=str, help="the elements contained in the database")
    parser.add_argument('--charge',    dest='charge',    default=0,                            type=int, help='total charge of the system (default: 0)')
    parser.add_argument('--spin',      dest='spin',      default=None,                         type=int, help='number of unpaired electrons (default: None) (use 0 to treat a closed-shell system in a UHF manner)')
    parser.add_argument('--xc',        dest='xc',        default=defaults.xc,                  type=str, help=f'DFT functional for the SAD guess (default: {defaults.xc})')
    parser.add_argument('--ecp',        dest='ecp',        default=None,                  type=str, help=f'effective core potential to use (default: None)')
    parser.add_argument('--nameout',   dest='NameOut',   default=None,                         type=str, help='name of the output representations file.')
    parser.add_argument('--omod',      dest='omod',      default=defaults.omod,     nargs='+', type=str, help=f'model(s) for open-shell systems (alpha, beta, sum, diff, default: {defaults.omod})')
    args = parser.parse_args()
    print(vars(args))

    mol = compound.xyz_to_mol(check_file(args.mol), args.basis, charge=args.charge, spin=args.spin, unit=args.units, ecp=args.ecp)
    dm = None if args.dm is None else np.load(args.dm)

    representations = get_repr(mol, elements, args.charge, args.spin,
                               open_mod=args.omod,
                               dm=dm, guess=args.guess, model=args.model,
                               xc=args.xc, auxbasis=args.auxbasis)

    # output dir
    cwd = os.getcwd()

    # save the output
    mol_name = args.mol.split('/')[-1].split('.')[0]
    if args.NameOut is not None:
        name_out = args.NameOut
    else:
        name_out = 'X_'+mol_name
    if args.spin is not None:
        name_out = name_out+'_'+'_'.join(args.omod)

    path_out = os.path.join(cwd, name_out)
    np.save(path_out, representations)

    print(f"Generated density-based representation for {mol_name} with")
    print("Type.\tlength")
    for q, v in representations:
        print(q+'\t'+str(len(v)))
    print(f"stored at : {path_out}\n")


if __name__ == '__main__':
    main()
