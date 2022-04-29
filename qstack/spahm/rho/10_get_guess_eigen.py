#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pyscf import scf, gto
from utils import readmol
from guesses import *
import lowdin
import operator
from Dmatrix import Dmatrix_for_z, c_split, rotate_c
import repre
import scipy

parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
parser.add_argument('--mol',    type=str,            dest='filename',  required=True,   help='file containing a list of molecular structures in xyz format')
parser.add_argument('--guess',  type=str,            dest='guess',     required=True,   help='initial guess type')
parser.add_argument('--basis',  type=str,            dest='basis'  ,   default='minao', help='AO basis set (default=MINAO)')
parser.add_argument('--charge', type=int,            dest='charge',    default=0,       help='total charge of the system (default=0)')
parser.add_argument('--spin',   type=int,            dest='spin',      default=None,    help='number of unpaired electrons (default=None) (use 0 to treat a closed-shell system in a UHF manner)')
parser.add_argument('--func',   type=str,            dest='func',      default='hf',    help='DFT functional for the SAD guess (default=HF)')
parser.add_argument('--dir',    type=str,            dest='dir',       default='./',    help='directory to save the output in (default=current dir)')
parser.add_argument('--cutoff', type=float,          dest='cutoff',    default=5.0,     help='bond length cutoff')
parser.add_argument('--zeros',  action='store_true', dest='zeros',     default=False,   help='if use a version with more padding zeros')
parser.add_argument('--split',  action='store_true', dest='split',     default=False,   help='if split into molecules')
args = parser.parse_args()
print(vars(args))

#1e-12: 6 A
#1e-10: 5 A
#1e-8 : 4 A
#1e-6 : 3 A

def mols_guess(xyzlist, basis, aguess):
  guess = get_guess(aguess)
  mols  = []
  dms   = []
  for xyzfile in xyzlist:
    mol = readmol(xyzfile, basis)
    if args.guess == 'huckel':
      e,v = scf.hf._init_guess_huckel_orbitals(mol)
    else:
      fock = guess(mol, args.func)
      e,v = solveF(mol, fock)
    mols.append( mol)
    dms.append( v[:,:mol.nelectron//2] @ v[:,:mol.nelectron//2].T )
  return mols, dms


def fit_dm(dm, mol, auxmol):
  e2c  = auxmol.intor('int2c2e_sph')
  pmol = mol + auxmol
  e3c  = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
  e3c  = e3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
  w    = np.einsum('pq,qpi->i', dm, e3c)
  c    = scipy.linalg.solve(e2c, w)
  return c

def bonds_dict_init(qqs, M):
  N = 0
  mybonds = {}
  for qq in qqs:
    n = len(M[qq])
    mybonds[qq] = np.zeros(n)
    N += n
  return mybonds, N

def make_aux_mol(ri0, ri1, mybasis):
  rm = (ri0+ri1)*0.5
  atom = "No  % f % f % f" % (rm[0], rm[1], rm[2])
  auxmol = gto.M(atom=atom, basis=mybasis)
  return auxmol


def vec_from_cs(z, cs, lmax, idx, M):
  D = Dmatrix_for_z(z, lmax)
  c_new = rotate_c(D, cs)
  v = repre.vectorize_c('No', idx, c_new)
  return v


def repr_for_mol(mol, dm, qqs, M, mybasis, idx):

  L = lowdin.Lowdin_split(mol, dm)
  q = [mol.atom_symbol(i) for i in range(mol.natm)]
  r = mol.atom_coords(unit='ANG')

  mybonds = [bonds_dict_init(qqs[q0], M) for q0 in q]
  maxlen = max([bond[1] for bond in mybonds])

  for i0 in range(mol.natm):
    for i1 in range(i0):
      q0, q1 = q[i0], q[i1]
      r0, r1 = r[i0], r[i1]
      z = r1-r0
      if np.linalg.norm(z)>args.cutoff:
        continue

      dm1 = L.get_bond(i0,i1)
      bname = operator.concat(*sorted((q0, q1)))
      auxmol = make_aux_mol(r0, r1, mybasis[bname])
      c  = fit_dm(dm1, mol, auxmol)
      cs = c_split(auxmol, c)
      lmax = max([ c[0] for c in cs])
      v0 = vec_from_cs(+z, cs, lmax, idx[bname], M[bname])
      v1 = vec_from_cs(-z, cs, lmax, idx[bname], M[bname])
      mybonds[i0][0][bname] += v0
      mybonds[i1][0][bname] += v1

  vec = [None]*mol.natm
  for i0 in range(mol.natm):
    vec[i0] = np.hstack([ M[qq] @ mybonds[i0][0][qq] for qq in qqs[q[i0]] ])
    vec[i0] = np.pad(vec[i0], (0, maxlen-len(vec[i0])), 'constant')
  return np.array(vec)

def get_element_pairs(elements):
  qqs0  = []
  qqs4q = {}
  for q1 in elements:
    qqs4q[q1] = []
    for q2 in elements:
      qq = operator.concat(*sorted((q1, q2)))
      qqs4q[q1].append(qq)
      qqs0.append(qq)
    qqs4q[q1].sort()
  qqs0 = sorted(set(qqs0))
  qqs = {}
  for q in elements:
    qqs[q] = qqs0
  return qqs, qqs4q

def read_df_basis(bnames):
  mybasis = {}
  for bname in bnames:
      if bname in mybasis: continue
      with open('basis/optimized/'+bname+'.bas', 'r') as f:
        mybasis[bname] = eval(f.read())
  return mybasis

def get_basis_info(qqs, mybasis):
  idx = {}
  M = {}
  for qq in qqs:
    print(qq)
    S, ao, _ = repre.get_S('No', mybasis[qq])
    idx[qq]  = repre.store_pair_indices_z(ao)
    M[qq]    = repre.metrix_matrix_z('No', idx[qq], ao, S)
  return idx, M

def read_basis_wrapper(mols):
  elements  = sorted(list(set([q for mol in mols for q in mol.elements])))
  qqs,qqs4q = get_element_pairs(elements)
  qqs0      = qqs[list(qqs.keys())[0]]
  mybasis   = read_df_basis(qqs0)
  idx, M    = get_basis_info(qqs0, mybasis)
  return elements, mybasis, qqs, qqs4q, idx, M


def main():

  xyzlistfile = args.filename
  xyzlist = repre.get_xyzlist(xyzlistfile)

  mols, dms = mols_guess(xyzlist, args.basis, args.guess)
  elements, mybasis, qqs0, qqs4q, idx, M = read_basis_wrapper(mols)
  qqs = qqs0 if args.zeros else qqs4q

  if args.split:
    natm   = max([mol.natm for mol in mols])
    maxlen = max([bonds_dict_init(qqs[q0], M)[1] for q0 in elements ])
    allvec = np.zeros((len(mols), natm, maxlen))
  else:
    allvec = []

  for i,(mol,dm) in enumerate(zip(mols,dms)):
    print('mol', i)
    vec = repr_for_mol(mol, dm, qqs, M, mybasis, idx)
    if args.split:
      allvec[i,:len(vec),:] = vec
    else:
      allvec.append(vec)

  if not args.split:
    allvec = np.vstack(allvec)


  print(allvec.shape)
  np.save('mygreatrepresentation', allvec)


if __name__ == "__main__":
  main()

