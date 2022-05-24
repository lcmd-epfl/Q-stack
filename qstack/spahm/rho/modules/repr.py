#!/bin/env python3

import numpy as np
from pyscf import gto, data

#np.set_printoptions(edgeitems=30, linewidth=100000)
# basis  = 'ccpvdzjkfit'
# method = 'sph-short'
#method = 'sph'

def get_xyzlist(xyzlistfile):
  xyzlist = np.loadtxt(xyzlistfile, dtype=str)
  if xyzlist.shape==(): return [str(xyzlist)]
  else:                 return list(xyzlist)

def mysqrtm(m):
  e,b = np.linalg.eigh(m)
  e[abs(e)<1e-13] = 0.0
  sm = b @ np.diag(np.sqrt(e)) @ b.T
  return (sm+sm.T)*0.5

def idxl0(i,l):
  # return the index of the basis function with the same L and N but M=0
  if l!=1:
    return i - ao[q]['m'][i]+l
  else:
    if ao[q]['m'][i]==1  : return i+2
    if ao[q]['m'][i]==-1 : return i+1
    if ao[q]['m'][i]==0  : return i

def read_mols(xyzlist, basis):
  mols  = []
  coefs = []
  for xyzfile in xyzlist:
    #print(xyzfile, flush=True)
    mol_name = xyzfile.split('/')[-1].split('.')[0]
    mol = pyscf_ext.readmol(xyzfile, basis)
    c = np.load('./'+c_dir+mol_name+'.c.npy')
    mols.append(mol)
    coefs.append(c)
    print(mol_name, flush=True)
  return mols, coefs

def get_S(q,basis):
  mol = gto.Mole()
  mol.atom = q + " 0.0 0.0 0.0"
  mol.charge = 0
  mol.spin = data.elements.ELEMENTS_PROTON[q]%2
  mol.basis = basis
  mol.build()
  S = mol.intor_symmetric('int1e_ovlp')

  i0 = 0
  ao = {'l':[], 'm':[]}
  ao_start = []
  for prim in mol._basis[q]:
    l = prim[0]
    msize = 2*l+1
    ao['l'].extend([l]*msize)
    if l!=1:
      ao['m'].extend(np.arange(msize)-l)
    else:
      ao['m'].extend([1,-1,0]) # x,y,z
    ao_start.append(i0)
    i0 += msize

  return S, ao, ao_start

def store_pair_indices(ao):
  idx = []
  for i,[li,mi] in enumerate(zip(ao['l'], ao['m'])):
    for j,[lj,mj] in enumerate(zip(ao['l'], ao['m'])):
      if (li!=lj) or (mi!=mj): continue
      idx.append([i,j])
  return idx

def store_pair_indices_short(ao, ao_start):
  idx = []
  for i in ao_start:
    for j in ao_start:
      li = ao['l'][i]
      lj = ao['l'][j]
      mi = ao['m'][i]
      mj = ao['m'][j]
      if li!=lj : continue
      idx.append([i,j])
  return idx

def metrix_matrix(q, idx, ao, S):
  N = len(idx)
  A = np.zeros((N,N))
  for p in range(N):
    for p1 in range(p,N):
      i,j  =idx[p]
      i1,j1=idx[p1]
      l  = ao['l'][i]
      l1 = ao['l'][i1]
      if(l!=l1): continue
      A[p1,p] = A[p,p1] = 1.0/(2*l+1) \
        * S[ idxl0(i, l), idxl0(i1, l) ] \
        * S[ idxl0(j, l), idxl0(j1, l) ]
  return mysqrtm(A)

def metrix_matrix_short(q, idx, ao, S):
  N = len(idx)
  A = np.zeros((N,N))
  for p in range(N):
    for p1 in range(p,N):
      i,j   = idx[p]
      i1,j1 = idx[p1]
      l  = ao['l'][i]
      l1 = ao['l'][i1]
      if(l!=l1): continue
      A[p1,p] = A[p,p1] = S[i,i1] * S[j,j1] / (2*l+1)
  return mysqrtm(A)

def vectorize_c(q, idx, c):
  v = np.zeros(len(idx))
  for p,(i,j) in enumerate(idx):
    v[p] = c[i]*c[j]
  return v

def vectorize_c_short(q, idx, ao, c):
  v = np.zeros(len(idx))
  for p,[i,j] in enumerate(idx):
    l = ao['l'][i]
    msize = 2*l+1
    v[p] = c[i:i+msize] @ c[j:j+msize]
  return v
