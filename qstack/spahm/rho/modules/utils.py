from genericpath import isfile
from posix import listdir
from numpy.lib.npyio import load, save
from pyscf import gto, scf, dft
import numpy as np
import os
import sys
import re
import pyscf
import scipy
from os.path import join, isfile, isdir
import resource
import time

from pyscf.gto import mole
from scipy.linalg.decomp_svd import subspace_angles
from scipy.linalg.special_matrices import block_diag


def error_handler(e):
    print('Error-Handler :', dir(e), sep=' ')
    print("-->{}<--".format(e.__cause__))

def unix_time_decorator(func):
# thanks to https://gist.github.com/turicas/5278558
  def wrapper(*args, **kwargs):
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    ret = func(*args, **kwargs)
    end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
    print(func.__name__, ':  real: %.4f  user: %.4f  sys: %.4f'%
          (end_time - start_time,
           end_resources.ru_utime - start_resources.ru_utime,
           end_resources.ru_stime - start_resources.ru_stime))
    return ret
  return wrapper


def standardize_vectors(X_array) :
    X_shape = X_array.shape
    X_std = np.zeros(X_shape)

    for v, vect in enumerate(X_array) :
        mean = np.mean(vect)
        std = np.std(vect)
        X_std[v] = (vect - mean)/std
    return X_std


def get_coeff_G(K_matrix, K_pool, Y_training, etha, var=1e-6, regul='+') :
    dim = len(K_pool)
    K_T = np.transpose(K_matrix)
    Y_comp  = np.matmul(K_T, Y_training)
    print(f"Using RegularizationMethod : {regul} !\n")
    if regul == '++' : # DOES NOT WORK
        dim = len(Y_training)
        gamma = np.identity(dim) * etha
        gamma_inv = np.identity(dim) * (1/etha)
        Y_comp = np.matmul(gamma_inv, Y_training)
        Y_comp  = np.matmul(K_T, Y_comp)
        K_regul = np.matmul(gamma_inv, K_matrix)
        K_regul = np.matmul(K_T, K_regul) + (etha * K_pool) + (np.identity(len(K_pool)) * var)
    elif regul == '+' : #works up to a given point
        K_regul = np.matmul(K_T, K_matrix) + (etha * K_pool) + (np.identity(dim) * var)
    elif regul == 'classic' : #does not work for sparsification ;; #TODO: try with square problem
#        print("K-matrix dim =", K_matrix.shape, sep=' ')
        K_regul = np.matmul(K_T, K_matrix) + (etha * K_pool)
#    K_invert = np.linalg.inv(K_regul)
#    coeffs = np.matmul(K_invert, Y_comp)
    coeffs = np.linalg.solve(K_regul, Y_comp)
#    np.save("K_" + str(dim) + "_sparse_" + str(regul), K_regul)
    return coeffs


def normalize_vectors(X_array) :
    for v, vect in enumerate(X_array) :
        v_norm = np.linalg.norm(vect)
        X_array[v, :] /= v_norm
    return X_array


def make_object(filename, basis_set) :
    mol = gto.Mole()
    mol.atom = filename
    mol.basis = basis_set
    mol.verbose = 0
    mol.build()
    return mol

def get_scf(molecule, theory, xc, initial_guess) :
    # if theory == 'hf' :
    #     scf_mol = scf.RHF(molecule)
    #     scf_mol.init_guess = initial_guess
    #     print(f"------------------Printing converged energy using {theory}-{initial_guess}------------------------\n")
    #     scf_mol.kernel()
    #     properties = {'mo_energy' : scf_mol.mo_energy, 'mo_coeff' : scf_mol.mo_coeff, 'e_tot' : scf_mol.e_tot, 'details' : '-'.join([theory, initial_guess, xc])};
    #     return properties
    # elif theory == 'dft' :
    scf_mol = dft.RKS(molecule)
    scf_mol.xc = xc
    pyscf.scf.get_init_guess
    scf_mol.init_guess = initial_guess
    scf_mol.max_cycle = -1
    scf_mol.kernel()
    print(f"\n\tinitial guess registered : {scf_mol.init_guess}\n")
    properties = {'mo_energy' : scf_mol.mo_energy, 'mo_coeff' : scf_mol.mo_coeff, 'e_tot' : scf_mol.e_tot, 'details' : '-'.join([theory, initial_guess, xc])};
    return properties


def quick_view(xyz_molecule) :
    with open(xyz_molecule, "r") as f :
        lines = f.readlines()
    print(f"Extract of the molecule-{xyz_molecule}-:\n{lines}\n")

def print_file_out(mol_file, results, dir_out=os.getcwd(), key_value='mo_energy') :
    if isinstance(results, np.ndarray) == False : results = np.array(results, dtype=object) ; print("converted!")
    dir_working = os.getcwd()
    # print(dir_working)
    dir_out = join(dir_working, dir_out)
    # print(dir_out)
    if isdir(dir_out) == False : os.makedirs(dir_out)
    reg_match = re.search(r'/([^/]+)\.xyz', mol_file)
    if reg_match : mol_name = reg_match.group(1)
    else : mol_name = 'UNKNOWN' ; print(f"Unrecognized filename ! FileOut saved at :\n\t{dir_out}/{mol_name}\n")
    print(f"FileOut saved at :\n\t{dir_out}/{mol_name}\n")
    np.save(f"{dir_out}/{mol_name}_{key_value}", results)

def get_overlap(molecule) :
    s = molecule.intor_symmetric('int1e_ovlp')
    return s

def get_hcore(molecule) :
    kin = molecule.intor_symmetric('int1e_kin')
    pot = molecule.intor_symmetric('int1e_nuc')
    h_approx = kin + pot
    # print(h_approx.shape)
    return h_approx

def get_e_c(molecule, h=None) :
    ovlp = get_overlap(molecule)
    if not isinstance(h, np.ndarray) : h = get_hcore(molecule)
    # else : h = load_H(molecule.atom, h)
    e, c = scipy.linalg.eigh(h, ovlp)
    return e, c

def get_dm(mo_coeff, mo_occ) :
    occ_coeff = mo_coeff[:,mo_occ > 0]
    # print(occ_coeff)
    dm = np.dot(occ_coeff*mo_occ[mo_occ > 0], occ_coeff.conj().T)
    return dm

def get_fit_integrals(molecule, aux_molecule) :
    # print("\t...Computing integrals")
    S = aux_molecule.intor('int1e_ovlp_sph')
    mol_aux_mol = molecule + aux_molecule
    eri2c = aux_molecule.intor('int2c2e_sph')
    eri3c = mol_aux_mol.intor('int3c2e_sph', shls_slice=(0, molecule.nbas, 0, molecule.nbas, molecule.nbas, molecule.nbas+aux_molecule.nbas))
    eri3c = eri3c.reshape(molecule.nao_nr(), molecule.nao_nr(), -1)
    return S, eri2c, eri3c

def get_fit_coeff(dm, eri2c, eri3c) :
    # print("\t...Computing Coefficients")
    rho = np.einsum('ijp, ij->p', eri3c, dm)
    coeff = np.linalg.solve(eri2c, rho)
    return coeff

def get_DF(mol, dm, basis='minao', aux_basis='ccpvdz-jkfit', out='print') :
    print(f"Computing Density-Fitting for molecule: {mol.atom}\n")
    print("Starting DF for\t"+mol.atom.split("/")[-1]+" ...")
    auxmol = make_object(mol.atom, 'ccpvdz-jkfit')
    aux_ovlp, eri2c, eri3c = get_fit_integrals(mol, auxmol)
    coefficients = get_fit_coeff(dm, eri2c, eri3c)
    print("...DF computed !\n")

    return coefficients

def get_shell_blocks(molecule, shell ='s') :
    shells = pyscf.gto.Mole.search_ao_label(molecule,f'.* .*{shell}')
    if len(shells) == 0 : return []
    blocks = []
    i = 0
    low = shells[i]
    if shell == 's' : largest = 0
    elif shell == 'p' : largest = 2
    elif shell == 'd' : largest = 4
    elif shell == 'f' : largest = 6
    while i < (len(shells)-1) :
        diff = np.abs(shells[i] - shells[i+1])
        if diff > 1 or np.abs(low - shells[i+1]) > largest :
            high = shells[i]
            blocks.append([low, high])
            i += 1
            low = shells[i]
        else : i += 1
    high = shells[i]
    blocks.append([low, high])
    return blocks

def sphericalize_H(mol, basis_set='minao', h_dir=None) :
    sub_shells = dict()
    print("Starting H-sphericalization for\t"+mol.split("/")[-1]+" ...")
    molecule = make_object(mol, basis_set)
    # molecule = molecule + molecule
    if h_dir == None : h = get_hcore(molecule)
    else : h = load_H(mol, h_dir)
    # print_file_out(mol, h, dir_out="Hamiltonian/minao", key_value='Hcore')
    init_shape = h.shape
    new_h = np.zeros(init_shape)
    print(f"Hamiltonian :\n{h}\n")
    print(f"Molecule's AO :\n{molecule.ao_labels()}\n")
    ao_labels = molecule.ao_labels()
    orbital_types = []
    for l in ao_labels :
        search = re.search(r'[0-9]+([a-z]).*', l)
        if search :
            found = search.group((1))
            orbital_types.append(found)
    orbital_types = set(orbital_types)
    for orbital in orbital_types :
        blocks = get_shell_blocks(molecule, shell=orbital)
        formatted = []
        for b in blocks :
            formatted.append([b, h[b[0]:b[1]+1, b[0]:b[1]+1]])
            sub_shells[orbital] = formatted
    for sub in sub_shells :
        # print(f"{sub}\n{sub_shells[sub]}")
        if sub == 's' :
            sub_block = sub_shells[sub]
            for b in sub_block :
                l = b[0][0]
                for j in sub_block :
                    c = j[0][1]
                    new_h[l, c] = h[l, c]
        else :
            sub_block = sub_shells[sub]
            for b in sub_block :
                l1 = b[0][0]
                l2 = b[0][1]
                for j in sub_block :
                    c1 = j[0][0]
                    c2 = j[0][1]
                    print(f"[{l1}:{l2},{c1}:{c2}]")
                    tmp_block = h[l1 : l2+1, c1 : c2+1]
                    n_avg = len(tmp_block[0][:])
                    tmp_trace = np.trace(tmp_block)/n_avg
                    averaged_block = np.ones((n_avg)) * tmp_trace
                    averaged_block = np.diag(averaged_block)
                    new_h[l1 : l2+1, c1 : c2+1] = averaged_block
    print("\t... Finished sphericalization : Ok !\n")
    print_file_out(mol, new_h, dir_out='LB/sph_DM', key_value='DM_sph')
    return new_h

def global_to_local(h, molecule) :
    atomic_slice = dict()
    pyscf.gto.Mole.atom_shell_ids
    mol_name = molecule.atom.split('/')[-1].split('.')[0]
    atomic_slice['name'] = mol_name
    atoms = molecule.elements
    print(f"Compouting atomic-local environments for mol : {mol_name}")
    a_slices = molecule.aoslice_by_atom()
    # print(a_slices)
    for i, s in enumerate(a_slices) :
        start = s[2]
        stop = s[3]
        h_slice = h[start:stop,:]
        if atoms[i] in atomic_slice.keys() : atomic_slice[atoms[i]].append(h_slice)
        else : atomic_slice[atoms[i]] = [h_slice]
    return atomic_slice

def load_H(molFile, dir_in) :
    current_dir = os.getcwd()
    working_dir = join(current_dir, dir_in)
    if isdir(working_dir) == False : print(f"No directory found at : {working_dir} ! \nExiting !!\n") ; exit()
    files_in = [join(working_dir, f) for f in listdir(working_dir) if isfile(join(working_dir, f))]

    molName = molFile.split('/')[-1]
    molName = molName.split('.')[-2]
    # for file in files_in :
    #     print(f"{molName} VS {file}\n{isfile(file)}")
    found = 0
    for file in files_in :
        # print(file)
        if molName in file :
            h = np.load(file)
            found += 1
    if found == 1 : return h
    elif found > 1 : print(f"Found too many matches for molecule with Name : {molName}\n\tin : {working_dir}")
    else : print(f"Found no matching molecule with Name : {molName}\n\tin : {working_dir}")

def dict_to_array(dict_list) :
    lengths = [len(d.keys()) for d in dict_list]
    max_l_id = lengths.index(max(lengths))
    atoms_symb = dict_list[max_l_id].keys()
    print(atoms_symb)
    n_atoms = len(atoms_symb)
    full_dict = dict()
    for atom in atoms_symb : full_dict[atom] = []
    local_store = []
    for d in dict_list :
        for atom in atoms_symb :
            if atom in d.keys() : full_dict[atom].append(d[atom])
            else : full_dict[atom].append([0])
    keys = full_dict.keys()
    for i, key in enumerate(full_dict) :
        local_store.append(full_dict[key])
    return local_store
