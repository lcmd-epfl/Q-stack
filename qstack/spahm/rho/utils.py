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
    elements=["H", "C", "N", "O", "S"]
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

def mols_guess(xyzlist, charge, spin, args):

  mols  = []
  for xyzfile,ch,sp in zip(xyzlist, charge, spin):
      if args.print>0: print(xyzfile, flush=True)
      mols.append(compound.xyz_to_mol(xyzfile, args.basis, charge=0 if ch is None else ch, spin=0 if ch is None else sp))
  if args.print>0: print()

  dms  = []

  if not args.readdm:
      guess = guesses.get_guess(args.guess)
      for xyzfile,mol in zip(xyzlist,mols):
          if args.print>0: print(xyzfile, flush=True)
          e,v = spahm.get_guess_orbitals(mol, guess)
          dm  = guesses.get_dm(v, mol.nelec, mol.spin if args.spin else None)
          dms.append(dm)
          if args.save:
              np.save(os.path.basename(xyzfile)+'.npy', dm)
  else:
      for xyzfile,mol in zip(xyzlist,mols):
          if args.print>0: print(xyzfile, flush=True)
          dm = np.load(args.readdm+'/'+os.path.basename(xyzfile)+'.npy')
          if args.spin and dm.ndim==3:
              dm = np.arrag((dm/2,dm/2))
          dms.append(dm)
  if args.print>0: print()

  return mols, dms



def dm_open_mod(dm, omod):
    dmmod = {'sum':   lambda dm: dm[0]+dm[1],
             'diff':  lambda dm: dm[0]-dm[1],
             'alpha': lambda dm: dm[0],
             'beta':  lambda dm: dm[1]}
    return dmmod[omod](dm)


def get_xyzlist(xyzlistfile):
  return np.loadtxt(xyzlistfile, dtype=str, ndmin=1)
