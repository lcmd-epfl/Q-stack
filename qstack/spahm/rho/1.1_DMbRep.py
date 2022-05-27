#!/usr/bin/env python3

import argparse
import os
from os.path import join, isfile, isdir
import numpy as np
import pyscf
from  qstack import compound, spahm
from modules import dmb_rep_atom as dmba

#temporary script to generate all the models

parser = argparse.ArgumentParser(description='Script intended for computing Density-Matrix based representations (DMbReps) for efficient quantum machine learning.')
parser.add_argument('--mol',       dest='pathToMol',    required=True,                       type=str,            help="The path to the xyz file with the molecular structure")
parser.add_argument('--guess',     dest='initialGuess', default='LB',                        type=str,            help="The initial guess Hamiltonian to be used. Default: LB")
parser.add_argument('--basis-set', dest='basisSet',     default='minao',                     type=str,            help="Basis set for computing density matrix (default : minao)")
parser.add_argument('--aux-basis', dest='auxBasisSet',  default='ccpvdzjkfit',               type=str,            help="Auxiliary basis set for density fitting the density-matrix (default: ccpdvz-jkfit")
parser.add_argument('--species',   dest='Species',      default = ["C", "H", "O", "N", "S"], type=str, nargs='+', help="The elements contained in the database")
args = parser.parse_args()
print(vars(args))

def main() :

    # User-defined options
    guess         = spahm.guesses.get_guess(args.initialGuess)
    basis_set     = args.basisSet
    aux_basis_set = args.auxBasisSet

    # Molecule-independent computation
    atom_types    = sorted(list(set(args.Species)))
    ao, ao_len, idx, M = dmba.get_basis_info(atom_types, aux_basis_set)

    mol_file = args.pathToMol
    mol = compound.xyz_to_mol(mol_file, basis_set)

    cwd = os.getcwd()
    dir_out = 'X_' + (cwd+'/'+mol_file).split('/')[-2]
    if not isdir(join(cwd, dir_out)) : os.mkdir(join(cwd, dir_out))
    mol_name = mol_file.split('/')[-1].split('.')[0]

    dm = spahm.compute_spahm.get_guess_dm(mol, guess)

    for mymodel in ['occup', 'pure', 'sad-diff', 'lowdin-short', 'lowdin-long', 'lowdin-short-x', 'lowdin-long-x']:
        model = dmba.get_model(mymodel)
        df_wrapper, sym_wrapper = model
        c_df    = df_wrapper(mol, dm, basis_set, aux_basis_set)
        vectors = sym_wrapper(c_df, mol, idx, ao, ao_len, M, atom_types)
        name_out = 'X_'+mol_name+'_'+mymodel
        path_out = join(cwd, dir_out, name_out)
        np.save(path_out, vectors)

if __name__ == '__main__' : main()

