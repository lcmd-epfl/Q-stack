#!/usr/bin/env python3

import argparse
import sys,os
from os.path import join, isfile, isdir
import numpy as np
import pyscf
from  qstack import compound, spahm, fields
import modules.repr as repre

from modules import make_atomic_DF

parser = argparse.ArgumentParser(description='Script intended for computing Density-Matrix based representations (DMbReps) for efficient quantum machine learning.')
parser.add_argument('--mol',       dest='pathToMol',    required=True,                       type=str,            help='The path to the xyz file with the molecular structure')
parser.add_argument('--guess',     dest='initialGuess', default='LB',                        type=str,            help="The initial guess Hamiltonian to be used. Default: LB")
parser.add_argument('--basis-set', dest='basisSet',     default='minao',                     type=str,            help="Basis set for computing density matrix (default : minao)")
parser.add_argument('--aux-basis', dest='auxBasisSet',  default='ccpvdzjkfit',               type=str,            help="Auxiliary basis set for density fitting the density-matrix (default: ccpdvz-jkfit")
parser.add_argument('--model',     dest='modelRep',     default='Lowdin',                    type=str,            help="The model to use when creating the representation. Available models : -Lowdin (default); -pure (simple DF(DM)); -SAD_diff")
parser.add_argument('--species',   dest='Species',      default = ["C", "H", "O", "N", "S"], type=str, nargs='+', help='The species contained in the database.')
args = parser.parse_args()
print(vars(args))

def get_basis_info(atom_types, aux_basis_set):
    ao = {}
    idx = {}
    M = {}
    ao_len = {}
    for q in atom_types:
        S, ao[q], ao_start = repre.get_S(q, aux_basis_set)
        idx[q] = repre.store_pair_indices_short(ao[q], ao_start)
        M[q]   = repre.metrix_matrix_short(q, idx[q], ao[q], S)
        ao_len[q] = len(S)
    return ao, ao_len, idx, M

def check_file(pathToMol):
    cwd = os.getcwd()                 #  TODO why?
    mol_file = join(cwd, pathToMol)   #  TODO why?
    if isfile(mol_file) == False :
        print(f"Error xyz-file {mol_file} not found !\nExiting !!", file=sys.stderr)
        exit(1)
    mol_name = mol_file.split('/')[-1].split('.')[0]
    return mol_file, mol_name, cwd

def get_model(arg):

    def df_pure(mol, dm, basis_set, aux_basis_set):
        return fields.decomposition.decompose(mol, dm, aux_basis_set)[1]
    def df_sad_diff(mol, dm, basis_set, aux_basis_set):
        mf = pyscf.scf.RHF(mol)
        dm_sad = mf.init_guess_by_atom(mol)
        dm = dm - dm_sad
        return fields.decomposition.decompose(mol, dm, aux_basis_set)[1]
    def df_lowdin(mol, dm, basis_set, aux_basis_set):
        return make_atomic_DF.get_a_DF(mol, dm , basis_set, aux_basis_set)

    arg = arg.lower()
    models = {'pure'    : [df_pure,     coefficients_symmetrize_short],
             'sad_diff' : [df_sad_diff, coefficients_symmetrize_short],
             'lowdin'   : [df_lowdin,   coefficients_symmetrize_long ] }
    if arg not in models.keys():
        print('Unknown model. Available models:', list(models.keys()), file=sys.stderr)
        exit(1)
    return models[arg]


def coefficients_symmetrize_short(c, mol, idx, ao, ao_len, M, _):
    # short lowdin / everything else
    v = []
    i0 = 0
    for q in mol.elements :
        n = ao_len[q]
        v.append([q, M[q] @ repre.vectorize_c_short(q, idx[q], ao[q], c[i0:i0+n])])
        i0 += n
    return v

def coefficients_symmetrize_long(c_df, mol, idx, ao, ao_len, M, atom_types):
    # long lowdin
    vectors = []
    for c_a, e in zip(c_df, mol.elements) :
        v_atom = {}
        for q in atom_types:
            v_atom[q] = np.zeros(len(idx[q]))
        i0 = 0
        for q in mol.elements :
            n = ao_len[q]
            v_atom[q] += M[q] @ repre.vectorize_c_short(q, idx[q], ao[q], c_a[i0:i0+n])
            i0 += n
        v_a = np.hstack([v_atom[q] for q in atom_types])
        vectors.append([e, v_a])
    return vectors

def main() :

    # User-defined options
    guess         = spahm.guesses.get_guess(args.initialGuess)
    model         = get_model(args.modelRep)
    basis_set     = args.basisSet
    aux_basis_set = args.auxBasisSet

    # Molecule-independent computation
    atom_types    = sorted(list(set(args.Species)))
    ao, ao_len, idx, M = get_basis_info(atom_types, aux_basis_set)

    # Generate compound from xyz file
    mol_file, mol_name, cwd = check_file(args.pathToMol)
    mol = compound.xyz_to_mol(mol_file, basis_set)

    # Compute density matrices
    print("Computing DM...")
    e,v = spahm.compute_spahm.get_guess_orbitals(mol, guess)
    dm = spahm.guesses.get_dm(v, mol.nelec, mol.spin)

    # Post-processing the density-matrix
    df_wrapper, sym_wrapper = model
    c_df    = df_wrapper(mol, dm, basis_set, aux_basis_set)
    vectors = sym_wrapper(c_df, mol, idx, ao, ao_len, M, atom_types)

    # save the output
    name_out = 'X_'+mol_name
    dir_out = 'X_' + mol_file.split('/')[-2]
    if not isdir(join(cwd, dir_out)) : os.mkdir(join(cwd, dir_out))
    path_out = join(cwd, dir_out, name_out)
    np.save(path_out, vectors)

    print(f"Generated density-based representation for {mol_name} with,\n")
    print("Type.\tlength")
    for q,v in vectors :
        print(q+'\t'+str(len(v)))
    print(f"stored at : {path_out}")


if __name__ == '__main__' : main()


