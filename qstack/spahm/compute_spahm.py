from pyscf import scf, grad
from qstack.spahm.guesses import solveF, get_guess, get_occ, get_dm, eigenvalue_grad

def get_guess_orbitals(mol, guess, xc="pbe"):
    if guess == 'huckel':
        e,v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
        fock = guess(mol, xc)
        e,v = solveF(mol, fock)
    return e,v

def get_guess_orbitals_grad(mol, guess):
    e, c = get_guess_orbitals(mol, guess[0])
    mf = grad.rhf.Gradients(scf.RHF(mol))
    s1 = mf.get_ovlp(mol)
    h1 = guess[1](mf)
    return eigenvalue_grad(mol, e, c, s1, h1)

def get_guess_dm(mol, guess, xc="pbe", openshell=None):
    e,v = get_guess_orbitals(mol, guess, xc)
    return get_dm(v, mol.nelec, mol.spin if mol.spin>0 or not openshell is None else None)

def get_spahm_representation(mol, guess_in, xc="pbe"):
    guess = get_guess(guess_in)
    e,v   = get_guess_orbitals(mol, guess, xc)
    e1    = get_occ(e, mol.nelec, mol.spin)
    return e1
