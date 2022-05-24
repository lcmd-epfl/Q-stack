import os
from os.path import join, isfile, isdir
from posix import listdir
import resource, time

import numpy as np
from pyscf import gto
import scipy

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


def make_object(filename, basis_set) :
    mol = gto.Mole()
    mol.atom = filename
    mol.basis = basis_set
    mol.verbose = 0
    mol.build()
    return mol




def get_overlap(molecule) :
    s = molecule.intor_symmetric('int1e_ovlp')
    return s


def get_e_c(molecule, h=None) :  ############################################### OK
    ovlp = get_overlap(molecule)
    if not isinstance(h, np.ndarray) : h = get_hcore(molecule)
    # else : h = load_H(molecule.atom, h)
    e, c = scipy.linalg.eigh(h, ovlp)
    return e, c

def get_dm(mo_coeff, mo_occ) :   ############################################### OK
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

def get_DF(mol, dm, basis='minao', aux_basis='ccpvdz-jkfit', out='print') :    # KEEP
    print(f"Computing Density-Fitting for molecule: {mol.atom}\n")
    print("Starting DF for\t"+mol.atom.split("/")[-1]+" ...")
    auxmol = make_object(mol.atom, 'ccpvdz-jkfit')
    aux_ovlp, eri2c, eri3c = get_fit_integrals(mol, auxmol)
    coefficients = get_fit_coeff(dm, eri2c, eri3c)
    print("...DF computed !\n")
    return coefficients

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

