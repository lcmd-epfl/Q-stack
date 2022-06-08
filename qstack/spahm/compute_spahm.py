from pyscf import scf
from qstack.spahm.guesses import solveF, get_guess, get_occ, get_dm

def get_guess_orbitals(mol, guess, xc="pbe"):
    if guess == 'huckel':
        e,v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
        fock = guess(mol, xc)
        e,v = solveF(mol, fock)
    return e,v

def get_guess_dm(mol, guess, xc="pbe", openshell=None):
    e,v = get_guess_orbitals(mol, guess, xc)
    return get_dm(v, mol.nelec, mol.spin if mol.spin>0 or not openshell is None else None)

def get_spahm_representation(mol, guess_in, xc="pbe"):
  guess = get_guess(guess_in)
  e,v   = get_guess_orbitals(mol, guess, xc)
  e1    = get_occ(e, mol.nelec, mol.spin)
  return e1

