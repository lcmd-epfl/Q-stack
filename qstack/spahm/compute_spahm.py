from pyscf import scf
from qstack.spahm.guesses import *

def get_spahm_representation(mol, guess_in, xc="pbe"):

  guess = get_guess(guess_in)


  if guess_in == 'huckel':
    e,v = scf.hf._init_guess_huckel_orbitals(mol)
  else:
    fock = guess(mol, xc)
    e,v = solveF(mol, fock)

  e1 = get_occ(e, mol.nelec, mol.spin)

  return e1
