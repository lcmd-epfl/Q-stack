#!/usr/bin/env python3

import argparse
from types import SimpleNamespace
import sys,os
from os.path import join, isfile, isdir
import numpy as np
import pyscf
from  qstack import compound, spahm
from .modules import utils, dmb_rep_atom as dmba


defaults = SimpleNamespace(
    initial_guess='LB',
    model='Lowdin-long-x',
    basis='minao',
    aux_basis='ccpvdzjkfit',
    omod=['alpha','beta']
  )


def check_file(mol_file):
    if isfile(mol_file) == False :
        print(f"Error xyz-file {mol_file} not found !\nExiting !!", file=sys.stderr)
        exit(1)
    return mol_file

def generate_ROHSPAHM(mol_file, atom_types, charge, spin, open_mod=defaults.omod, dm=None, guess=defaults.initial_guess, model=defaults.model, basis_set=defaults.basis, aux_basis_set=defaults.aux_basis):

    # User-defined options
    guess         = spahm.guesses.get_guess(guess)
    model         = dmba.get_model(model)
    df_wrapper, sym_wrapper = model
    ao, ao_len, idx, M = dmba.get_basis_info(atom_types, aux_basis_set)

    # Generate compound from xyz file
    check_file(mol_file)
    mol = compound.xyz_to_mol(mol_file, basis_set, charge=charge, spin=spin)

    # Compute density matrices
    print("Computing DM...")
    if dm == None:
        dm = spahm.compute_spahm.get_guess_dm(mol, guess, openshell=spin)

    rep = []
    for omod in open_mod:
        if not spin:
            DM = dm
        else:
            DM = utils.dm_open_mod(dm, omod)

        # Post-processing of the density matrix
        c_df    = df_wrapper(mol, DM, basis_set, aux_basis_set)
        vectors = sym_wrapper(c_df, mol, idx, ao, ao_len, M, atom_types)
        rep.append(vectors)

        if spin == None:
            rep = vectors
            break
    rep = np.array(rep, dtype=object)
    return rep



def main() :

    parser = argparse.ArgumentParser(description='Script intended for computing Density-Matrix based representations (DMbReps) for efficient quantum machine learning.')
    parser.add_argument('--mol',       dest='pathToMol',    required=True,                       type=str,            help="The path to the xyz file with the molecular structure")
    parser.add_argument('--guess',     dest='initialGuess', default='LB',                        type=str,            help="The initial guess Hamiltonian to be used. Default: LB")
    parser.add_argument('--basis-set', dest='basisSet',     default='minao',                     type=str,            help="Basis set for computing density matrix (default : minao)")
    parser.add_argument('--aux-basis', dest='auxBasisSet',  default='ccpvdzjkfit',               type=str,            help="Auxiliary basis set for density fitting the density-matrix (default: ccpdvz-jkfit")
    parser.add_argument('--model',     dest='modelRep',     default='Lowdin-long-x',             type=str,            help="The model to use when creating the representation" )
    parser.add_argument('--dm',        dest='dm',           default=None,                        type=str,            help="The density matrix to load instead of computing the guess" )
    parser.add_argument('--species',   dest='Species',      default = ["C", "H", "O", "N", "S"], type=str, nargs='+', help="The elements contained in the database")
    parser.add_argument('--charge',    dest='charge',       default=0,                           type=int,            help='total charge of the system (default=0)')
    parser.add_argument('--spin',      dest='spin',         default=None,                        type=int,            help='number of unpaired electrons (default=None) (use 0 to treat a closed-shell system in a UHF manner)')
    parser.add_argument('--out',      dest='NameOut',         default=None,                        type=str,            help='nome of the outpute representations file.')
    parser.add_argument('--omod',      type=str,            dest='omod',      default=['alpha','beta'], nargs='+',  help='model for open-shell systems (alpha, beta, sum, diff')
    args = parser.parse_args()
    print(vars(args))


    # Generate list of species account for in the representation
    atom_types    = sorted(list(set(args.Species)))

    # Genereate atomic local SPAHM representation
    representations = generate_ROHSPAHM(args.pathToMol, atom_types, args.charge, args.spin, dm=None, guess=args.initialGuess, model=args.modelRep, basis_set=args.basisSet, aux_basis_set=args.auxBasisSet)

    # output dir
    cwd = os.getcwd()
    dir_out = 'X_' + (cwd+'/'+args.pathToMol).split('/')[-2]
    if not isdir(join(cwd, dir_out)) : os.mkdir(join(cwd, dir_out))



    # save the output
    mol_name = args.pathToMol.split('/')[-1].split('.')[0]

    for omod, vectors in zip(args.omod, representations):
        if args.NameOut != None:
            name_out = args.NameOut
        else:
            name_out = 'X_'+mol_name
        if not args.spin:
            name_out = name_out+'_'+omod
        path_out = join(cwd, dir_out, name_out)
        np.save(path_out, vectors)
        if args.spin is None:
            break

    print(f"Generated density-based representation for {mol_name} with")
    print("Type.\tlength")
    for q,v in vectors :
        print(q+'\t'+str(len(v)))
    print(f"stored at : {path_out}\n")



if __name__ == '__main__' : main()


