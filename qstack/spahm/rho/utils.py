import sys, os
import numpy as np
from types import SimpleNamespace
import qstack.spahm.compute_spahm as spahm
import qstack.spahm.guesses as guesses
from qstack import compound

defaults = SimpleNamespace(
    guess='LB',
    model='Lowdin-long-x',
    basis='minao',
    auxbasis='ccpvdzjkfit',
    omod=['alpha', 'beta'],
    elements=["H", "C", "N", "O", "S"],
    cutoff=5.0,
    xc='hf',
    bpath=os.path.dirname(__file__)+'/basis_opt',
  )



def get_chsp(f, n):
    if f:
      chsp = np.loadtxt(f, dtype=int).reshape(-1)
      if(len(chsp)!=n):
          print('Wrong lengh of the file', f, file=sys.stderr);
          exit(1)
    else:
        chsp = np.zeros(n, dtype=int)
    return chsp

def load_mols(xyzlist, charge, spin, basis, printlevel=0):
    mols = []
    for xyzfile, ch, sp in zip(xyzlist, charge, spin):
        if printlevel>0: print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis, charge=0 if ch is None else ch, spin=0 if ch is None else sp)) #TODO
    if printlevel>0: print()
    return mols

def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=False, printlevel=0):
    dms = []
    guess = guesses.get_guess(guess)
    for xyzfile, mol in zip(xyzlist, mols):
        if printlevel>0: print(xyzfile, flush=True)
        if not readdm:
            e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm   = guesses.get_dm(v, mol.nelec, mol.spin if spin else None)
        else:
            dm = np.load(readdm+'/'+os.path.basename(xyzfile)+'.npy')
            if spin and dm.ndim==2:
                dm = np.array((dm/2,dm/2))
        dms.append(dm)
        if printlevel>0: print()
    return dms


def dm_open_mod(dm, omod):
    dmmod = {'sum':   lambda dm: dm[0]+dm[1],
             'diff':  lambda dm: dm[0]-dm[1],
             'alpha': lambda dm: dm[0],
             'beta':  lambda dm: dm[1]}
    return dmmod[omod](dm)


def get_xyzlist(xyzlistfile):
  return np.loadtxt(xyzlistfile, dtype=str, ndmin=1)
