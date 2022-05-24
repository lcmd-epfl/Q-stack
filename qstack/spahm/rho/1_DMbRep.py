#!/usr/bin/env python3

import numpy as np
from modules.make_atomic_DF import get_a_DF
from modules.utils import *
import argparse
from os.path import join, isfile, isdir
import os
from modules.repr import *
from modules.pyscf_ext import *
import pyscf
from pyscf import gto
from modules.LB2020guess import LB2020guess


parser = argparse.ArgumentParser(description='Script intended for computing Density-Matrix based representations (DMbReps) for efficient quantum machine learning.')
parser.add_argument('--mol', dest='pathToMol', required=True, type=str, help='The path to the molecular file within the database (must be a FILE)')
parser.add_argument('--guess', dest='initialGuess', required=False, default='LB', type=str, help="The initial guess Hamiltonian to be used."\
                                                                                    "Can be of the following :\n"\
                                                                                    "- Hcore (default)\n- ANO\n- SAD\n-Huckel\n- LB")
parser.add_argument('--basis-set', dest='basisSet',  required=False, default='minao', type=str, help="Basis set for computing density matrix (default : minao)")
parser.add_argument('--aux-basis', dest='auxBasisSet', required=False, default='ccpvdzjkfit', type=str, help="auxiliary basis-set for density-fitting the density-matrix (default: ccpdvz-jkfit")
parser.add_argument('--model', dest='modelRep', required=False, default='Lowdin', type=str, help="The model to use when creating the representation.\n"\
                                                                                "Availabel model :\n"\
                                                                                "- Lowdin (default)\n- pure (simple DF(DM))\n- SAD_diff")
parser.add_argument('--species', required = False, type = str, nargs='+', dest='Species', help = 'The species contained in the DatBase.')

args = parser.parse_args()

"""
TO RESOLVE :
    - make the script molecule or DB based ??? --> Make molecular for now ! ; maybe extend after ??
    - to parallelize density-matrix computations ?
    - including H-LB_script within utils.py or differently ? DONE : included as side-script
    - should we save temporary objects (hamiltonina , dm) ? or make optional (-debug-) ??
"""

def LB(mol):
    return LB2020guess(parameters='HF').Heff(mol)

def main() :
# User-defined options (initial guess ; basis-set ; auxiliary basis-set ; model)
    guesses = ['Hcore', 'Huckel', 'ANO', 'LB']
    if args.initialGuess not in guesses :
        print("input-error : guess not recognized --> using default (LB) !\n")
        Hguess = 'LB'
    else :
        Hguess = args.initialGuess
    basis_set = args.basisSet
    models = ['pure', 'SAD_diff', 'Lowdin']
    if args.modelRep not in models :
        print("input-error : model not recognized --> using default (Lowdin) !\n")
        model = 'Lowdin'
    else :
        model = args.modelRep
    aux_basis_set  = args.auxBasisSet
    params = {'guess' : Hguess, 'basis' : basis_set, 'model' : model, 'aux_basis' : aux_basis_set}
    print(f"User-defined parameters : {params}\n")


# Generating compound from xyz file
    cwd = os.getcwd()
    mol_file = join(cwd, args.pathToMol)
    if isfile(mol_file) == False :
        print(f"Error xyz-file {mol_file} not found !\nExiting !!")
        exit()
    mol = make_object(mol_file, basis_set)
    mol_name = mol_file.split('/')[-1].split('.')[0]

    if args.Species :
        atom_types = sorted(list(set(args.Species)))
    else :
        atom_types = sorted(list(set(["C", "H", "O", "N", "S"])))
    print(atom_types)

# Generating Hamiltonian matrices (if needed cf. LB) and computing density matrices
    mf = pyscf.scf.RHF(mol)
    print("Computing DM...")
    if Hguess == 'Hcore' :
        dm = mf.init_guess_by_1e(mol)
    elif Hguess == 'ANO' :
        dm = mf.init_guess_by_minao(mol)
    elif Hguess == 'Huckel' :
        dm = mf.init_guess_by_huckel(mol)
    elif Hguess == 'LB' :
        h = LB(mol)
        e, coeffs = get_e_c(mol, h=h)
        occ = mol.get_occ(e,coeffs)
        dm = get_dm(coeffs, occ)

# Post-processing the density-matrix
# Density-Fitting

    if model == 'pure' :
        c_df = get_DF(mol, dm, basis=basis_set, aux_basis=aux_basis_set)
    elif model == 'Lowdin' :
        c_df = get_a_DF(mol, dm , basis_set, aux_basis_set)
    elif model == 'SAD_diff' :
        dm_sad = mf.init_guess_by_atom(mol)
        dm = dm - dm_sad
        c_df = get_DF(mol, dm, basis=basis_set, aux_basis=aux_basis_set)

# Power-Spectrum generation --> make function // one for SAD & pure-DM AND one for Lowind / new-lowdin
# TODO: Modify computations according to new-lowdin ; check 11_DMBRep_DB.py for suitable modifications (carefull with padding)



    elements = sorted(list(set(mol.elements)))

    S  = {}
    ao = {}
    ao_start = {}
    idx = {}
    M = {}

    for q in atom_types:
      S[q], ao[q], ao_start[q] = get_S(q, aux_basis_set)
      idx[q] = store_pair_indices_short(ao[q], ao_start[q])
      M[q] = metrix_matrix_short(q, idx[q], ao[q], S[q])





    vectors = []
    for c_a, e in zip(c_df, mol.elements) :

        # KSENIA'S OPTIMIZED CODE !!!!

            v_atom = {}

            for q in atom_types:
                v_atom[q] = np.zeros(len(idx[q]))

            i0 = 0
            for q in mol.elements :
                n_size = len(S[q])
                v_atom[q] += M[q] @ vectorize_c_short(q, idx[q], ao[q], c_a[i0:i0+n_size])
                i0 += n_size


            v_a = np.hstack([v_atom[q] for q in atom_types])
            vectors.append([e, v_a])


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


