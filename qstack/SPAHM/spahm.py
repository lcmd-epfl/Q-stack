import os
import argparse
import numpy as np
from pyscf import scf
from qstack import compound
from guesses import *

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
parser.add_argument('--mol',    type=str, dest='filename', required=True,   help='path to molecular structure in xyz format')
parser.add_argument('--guess',  type=str, dest='guess',    required=True,   help='initial guess type')
parser.add_argument('--basis',  type=str, dest='basis'  ,  default='minao', help='AO basis set (default=MINAO)')
parser.add_argument('--charge', type=int, dest='charge',   default=0,       help='total charge of the system (default=0)')
parser.add_argument('--spin',   type=int, dest='spin',     default=None,    help='number of unpaired electrons (default=None) (use 0 to treat a closed-shell system in a UHF manner)')
parser.add_argument('--func',   type=str, dest='func',     default='hf',    help='DFT functional for the SAD guess (default=HF)')
parser.add_argument('--dir',    type=str, dest='dir',      default='./',    help='directory to save the output in (default=current dir)')
args = parser.parse_args()

def spahm(mol, guess_in):

  guess = get_guess(guess_in)


  if args.guess == 'huckel':
    e,v = scf.hf._init_guess_huckel_orbitals(mol)
  else:
    fock = guess(mol, args.func)
    e,v = solveF(mol, fock)

  e1 = get_occ(e, mol.nelec, args.spin)

  np.save(args.dir+'/'+args.guess+'_'+args.basis+'_'+name, e1)

if __name__ == "__main__":
  main()
