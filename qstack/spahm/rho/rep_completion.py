#!/usr/bin/env python3

import argparse
from types import SimpleNamespace
import sys,os
from os.path import join, isfile, isdir
import numpy as np
import pyscf
from  qstack import compound
import qstack.tools as tls
from . import utils, dmb_rep_atom as dmba, dmb_rep_bond as dmbb



def print_elem(vector,  pattern, atoms, extract=False):
    if type(atoms) != list:
        atoms = [atoms]
    if len(atoms) == 0 :
        print("Please select the desired atomic fragment, try again!\nExiting!")
        exit(1)
    for atom in atoms:
        if atom not in pattern.keys() :
            print(f"{atom} not present in feature vector pattern !\nExiting!")
            exit(1)
        print("Chosen atom: ", atom, end=' '*10)
        print("Vector fragment length =", len(vector[pattern[atom][0]:pattern[atom][1]]))
        print("Norm = ", np.linalg.norm(vector[pattern[atom][0]:pattern[atom][1]]))
        if extract:
            print("Extract:\n", vector[pattern[atom][0]:pattern[atom][1]])
        if len(atoms) == 1:
            return np.linalg.norm(vector[pattern[atom][0]:pattern[atom][1]])

def transform(vector, source_atoms, dest_atoms, aux_basis_set, atom_id=None):
    source_atoms = sorted(source_atoms)
    dest_atoms = sorted(dest_atoms)
    source_idx = get_vpattern(source_atoms, aux_basis_set)
    dest_idx = get_vpattern(dest_atoms, aux_basis_set)
    dest_len = np.array(list(dest_idx.values())).flatten().max()
    dest_vector = np.zeros((dest_len, ))
    for k, idx in dest_idx.items():
        if k in source_idx.keys():
            source_start = source_idx[k][0]
            source_stop = source_idx[k][1]
            dest_vector[idx[0]:idx[1]] += vector[source_start:source_stop]
#    return dest_vector if atom_id == None else [atom_id, dest_vector]
    if atom_id == None:
        return dest_vector
    else:
        return atom_id, dest_vector

def transform_bond(atom_type, vector, source_bonds, dest_bonds, bpath):
    print(atom_type)
    source_idx, source_len = get_vpattern_bond(source_bonds, bpath)
    dest_idx, dest_len = get_vpattern_bond(dest_bonds, bpath)
    dest_vector = np.zeros((dest_len, ))
    for k, idx in dest_idx[atom_type].items():
        if k in source_idx[atom_type].keys():
            source_start = source_idx[atom_type][k][0]
            source_stop = source_idx[atom_type][k][1]
            dest_vector[idx[0]:idx[1]] += vector[source_start:source_stop]
#    return dest_vector if atom_id == None else [atom_id, dest_vector]
    return atom_type, dest_vector

def get_vpattern(atom_set, aux_basis_set):
    atom_set = sorted(atom_set)
    ao, ao_len, idx, _= dmba.get_basis_info(atom_set, aux_basis_set)
    v_feat = dict()
    total = 0
    for k in atom_set:
        start = total
        total += len(idx[k])
        v_feat[k] = [start, total]
    return v_feat

def get_vpattern_bond(bonds_dict, bpath, omod=False, same_basis=False):
    bond_pairs = []
    for q, qq in bonds_dict.items():
        bond_pairs.extend(qq)
    bond_pairs = set(bond_pairs)
    mybasis = dmbb.read_df_basis(bond_pairs, bpath, same_basis=same_basis)
    idx, M  = dmbb.get_basis_info(bond_pairs, mybasis, False, 0)
    qq_sizes = {qq:len(idx[qq]) for qq in idx.keys()}
    max_bonds = max([len(val) for val in bonds_dict.values()])
    max_length = max(qq_sizes.values()) * max_bonds
    v_atoms = {}
    for q, qq in bonds_dict.items():
        v_feat = {}
        start = 0
        for iqq in qq:
            end = start +  qq_sizes[iqq]
            v_feat[iqq] =  [start, end]
            start = end
        v_atoms[q] = v_feat
        if omod:
            v_beta[q] = {k:[val[0]+end, val[1]+end] for k, val in v_atoms[q].items()}
    if omod:
        return v_atoms, v_beta, max_length*2
    else:
        return v_atoms, max_length

def test_equivalence(old_vectors, new_vectors):
    correspondances = []
    for nv in new_vectors[:,1]:
        l_corr = []
        for ov in old_vectors[:,1]:
            l_corr.append(np.linalg.norm(nv) == np.linalg.norm(ov))
        correspondances.append(l_corr)
    print([sum(loc) for loc in correspondances])
    return 0

_func=None
def worker_init(func):
    global _func
    _func = func 

def worker(*args):
    return _func(*args)

def get_transformer(mode):
    if mode == 'bond':
        return lambda v, rin, rout, bp, q : transform_bond(q, v, rin, rout, bp)
    elif mode == 'atom':
        return lambda v, rin, rout, aux, q : transform(v, rin, rout, aux, atom_id=q)
    else:
        print('Mode : '+mode+' not known ! Exiting.')
        exit(1)


def fromr1tor2(vectors, r1_info, r2_info, atom_type=None, bond=False, bpath=None, aux_basis=None, multi=False):
    tls.correct_num_threads()

    new_vectors = []
    if bond :
        if bpath == None:
            print("Missing atom_type or bond-basis path for bond-representation conversion")
            exit(1)
        mode = 'bond'
        aux = bpath
    else:
        aux = aux_basis
        mode = 'atom'
    mytransform = get_transformer(mode)
    if multi:
        import multiprocessing as mp
        count = len(os.sched_getaffinity(0))
        pool = mp.Pool(count, initializer=worker_init, initargs=(mytransform,))
        print(f"Entering parrallel mode [Using {count} cores] !", flush=True)

        def collect(result):
            new_vectors.extend(result)
        def collect_error(result):
            print(result)

        results = pool.starmap_async(worker, [(v, r1_info, r2_info, aux, a) for a, v in vectors], callback=collect, error_callback=collect_error)
        pool.close()
        pool.join()


    else:
        for atom, v in vectors:
            new_vec = mytransform(v, r1_info, r2_info, aux, atom)
            new_vectors.append([atom, new_vec])
    new_vectors= np.array(new_vectors, dtype=object, like=vectors)
    return new_vectors



EXPLORATION_EXAMPLE=False
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to convert feature vector from one set of atom types to another.')
    parser.add_argument('--X', type = str, dest='XREP', required=True, help='A txt file with the list of paths to the representation file (.npy)')
    parser.add_argument('--basis', type = str, dest='BasisSet', required=True, help='A txt file with the list of paths to the representation file (.npy)')
    parser.add_argument('--source', type = str, nargs= '+', dest='AtomsIn', required=True, help='The set of atom types the representation were generated with.')
    parser.add_argument('--destination', type = str, nargs= '+', dest='AtomsOut', required=True, help='The target set of atom types for the new feature vector.')
    parser.add_argument('--fout', type = str, dest='FileOut', required=False, default=None, help='The ouput filename or path.')
    parser.add_argument('--multi', action='store_true', dest='PARA', required=False, help='Use OMP parrallelization or not.')

    args = parser.parse_args()

    vectors = np.load(args.XREP, allow_pickle=True)
    new_vectors = []

    tranformers = {'atom': transform, 'bond': transform_bond}



    if args.PARA:
        import multiprocessing as mp
        count = mp.cpu_count()
        pool = mp.Pool(count)
        print(f"Entering parrallel mode [Using {count} cores] !")

        def collect(result):
            new_vectors.extend(result)

        results = pool.starmap_async(transform, [(v, args.AtomsIn, args.AtomsOut, args.BasisSet, a) for a, v in vectors], callback=collect)
        pool.close()
        pool.join()


    else:
        for atom, v in vectors:
            new_vec = transform(v, args.AtomsIn, args.AtomsOut, args.BasisSet)
            new_vectors.append([atom, new_vec])
    new_vectors= np.array(new_vectors, dtype=object, like=vectors)
    np.save(args.FileOut, new_vectors, allow_pickle=True)


    if EXPLORATION_EXAMPLE:
        vectors = np.load('/home/calvino/yannick/SPAHM-RHO/rxn/Hydroform/X_Hydro/rh/r/XR_Cat100.npy', allow_pickle=True)
        print([[atom, idx] for idx, atom in enumerate(vectors[:5,0])])
        atom_types = ['F', 'H', 'Rh', 'C', 'Cl', 'O', 'P']
        dest_types = ['F', 'H', 'Rh', 'C', 'Cl', 'O', 'P', 'Ir']
        atom_types = sorted(atom_types)
        aux_basis_set = 'def2svpjkfit'
        ao, ao_len, idx, M = dmba.get_basis_info(atom_types, aux_basis_set)
        print(len(ao['H']['m']),len(ao['H']['l']), ao_len.keys(), idx['H'][10], sep='\n')

        v_feat = get_vpattern(atom_types, aux_basis_set)
        print(v_feat.items())

        C = []
        i=0
        for a, v in vectors:
            if a == 'C' :
                print(i, end='    ')
                C.append([i, print_elem(v,  v_feat, 'O')])
            i+=1
        C = np.array(C)
        print(C.shape)
        max_idx = np.argmax(C[:,1])
        print("Max Norm = ", C[:,1].max(), " ; idx = ", C[max_idx,0])
        print_elem(vectors[16,1], v_feat, atom_types)
        vec = vectors[0,1]
        print(vec.shape)
        vec = transform(vec, atom_types, dest_types, aux_basis_set)
        print(vec.shape)
        dest_feat = get_vpattern(dest_types, aux_basis_set)
        print_elem(vec,  dest_feat, 'Ir', extract=True)

    return 0




if __name__ == '__main__' : main()
