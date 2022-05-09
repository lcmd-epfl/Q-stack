#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import gto,data
import pyscf_ext

#np.set_printoptions(edgeitems=30, linewidth=100000)
basis  = 'ccpvdzjkfit'
method = 'sph-short'
#method = 'sph'
method = 'z'

def get_xyzlist(xyzlistfile):
  xyzlist = np.loadtxt(xyzlistfile, dtype=str)
  if xyzlist.shape==(): return [str(xyzlist)]
  else:                 return list(xyzlist)

def mysqrtm(m):
  e,b = np.linalg.eigh(m)
  e[abs(e)<1e-13] = 0.0
  sm = b @ np.diag(np.sqrt(e)) @ b.T
  return (sm+sm.T)*0.5

def idxl0(i,l, ao):
  # return the index of the basis function with the same L and N but M=0
  if l!=1:
    return i - ao['m'][i]+l
  else:
    if ao['m'][i]==1  : return i+2
    if ao['m'][i]==-1 : return i+1
    if ao['m'][i]==0  : return i

def read_mols(xyzlist, basis):
  mols  = []
  coefs = []
  for xyzfile in xyzlist:
    #print(xyzfile, flush=True)
    mol = pyscf_ext.readmol(xyzfile, basis)
    c = np.load(xyzfile+'.c.npy')
    mols.append(mol)
    coefs.append(c)
  return mols, coefs

def get_S(q, basis):
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
        * S[ idxl0(i, l, ao[q]), idxl0(i1, l, ao[q]) ] \
        * S[ idxl0(j, l, ao[q]), idxl0(j1, l, ao[q]) ]
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

def store_pair_indices_z(ao):
  idx = []
  for i,[li,mi] in enumerate(zip(ao['l'], ao['m'])):
    for j,[lj,mj] in enumerate(zip(ao['l'], ao['m'])):
      if abs(mi)!=abs(mj): continue
      idx.append([i,j])
  return idx

def store_pair_indices_z_only0(ao):
  idx = []
  for i,[li,mi] in enumerate(zip(ao['l'], ao['m'])):
    if mi!=0 : continue
    for j,[lj,mj] in enumerate(zip(ao['l'], ao['m'])):
      if mj!=0 : continue
      idx.append([i,j])
  return idx


def metrix_matrix_z(q, idx, ao, S):
  N = len(idx)
  A = np.zeros((N,N))
  for p in range(N):
    for p1 in range(p,N):
      i,i1 = idx[p]
      j,j1 = idx[p1]
      li  = ao['l'][i ]
      li1 = ao['l'][i1]
      lj  = ao['l'][j ]
      lj1 = ao['l'][j1]
      if li  != lj  : continue
      if li1 != lj1 : continue

      mi  = ao['m'][i ]
      mi1 = ao['m'][i1]
      mj  = ao['m'][j ]
      mj1 = ao['m'][j1]

      A[p1,p] = A[p,p1] = ( (mi==mj)*(mi1==mj1) +  (mi==-mj)*(mi1==-mj1)*(mi!=0) ) \
        * S[ idxl0(i,  li , ao), idxl0(j,  li , ao)  ] \
        * S[ idxl0(i1, li1, ao), idxl0(j1, li1, ao) ]


  return mysqrtm(A)

################################################################################

def main():
  xyzlistfile = sys.argv[1]
  xyzlist = get_xyzlist(xyzlistfile)
  mols, coefs = read_mols(xyzlist, basis)

  elements = sorted(list(set([q for mol in mols for q in mol.elements])))

  S  = {}
  ao = {}
  ao_start = {}
  idx = {}
  M = {}

  for q in elements:
    S[q], ao[q], ao_start[q] = get_S(q, basis)
    if method == 'sph':
      idx[q]     = store_pair_indices(ao[q])
      M[q] = metrix_matrix(q, idx[q], ao[q], S[q])
    elif method == 'sph-short':
      idx[q] = store_pair_indices_short(ao[q], ao_start[q])
      M[q] = metrix_matrix_short(q, idx[q], ao[q], S[q])
    elif method == 'z':
      idx[q]     = store_pair_indices_z(ao[q])
      M[q] = metrix_matrix_z(q, idx[q], ao[q], S[q])

  vectors = []
  for mol,c in zip(mols, coefs):
    vectors.append([])
    c_split = []
    i0 = 0
    for q in mol.elements:
      n = len(S[q])
      c_at = c[i0:i0+n]
      i0 += n

      if method == 'sph':
        v = vectorize_c(q, idx[q], c_at)
      elif method == 'sph-short':
        v = vectorize_c_short(q, idx[q], ao[q], c_at)
      elif method == 'z':
        v = vectorize_c(q, idx[q], c_at)
      v = M[q] @ v
      vectors[-1].append(v)

  #for i in vectors:
  #    for j in i:
  #        print(j)
  #
  np.save('1.v.dat', vectors[0][0])
  np.save('2.v.dat', vectors[1][0])


if __name__=='__main__':
    main()
