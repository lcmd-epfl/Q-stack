#!/usr/bin/env python3
import sys
import numpy as np
from pyscf import dft,df,tools
import pyscf_ext

basis  = 'minao'

molfile = sys.argv[1]
charge  = int(sys.argv[2])
spin    = int(sys.argv[3])
at1idx  = int(sys.argv[4])-1
at2idx  = int(sys.argv[5])-1

def mysqrtm(m):
  e,b = np.linalg.eigh(m)
  e   = np.sqrt(e)
  sm  = b @ np.diag(e    ) @ b.T
  sm1 = b @ np.diag(1.0/e) @ b.T
  return (sm+sm.T)*0.5, (sm1+sm1.T)*0.5

mol = pyscf_ext.readmol(molfile, basis, charge=charge, spin=spin)

mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.run()


S = mol.intor_symmetric('int1e_ovlp')
S12,S12i = mysqrtm(S)

dm = mf.make_rdm1()
if len(dm.shape)==3:
  dm = dm[0]+dm[1]

dmL = S12  @ dm  @ S12

mo1idx = range(*mol.aoslice_nr_by_atom()[at1idx][2:])
mo2idx = range(*mol.aoslice_nr_by_atom()[at2idx][2:])
ix1 = np.ix_(mo1idx,mo2idx)
ix2 = np.ix_(mo2idx,mo1idx)

dmL_bond = np.zeros_like(dmL)
dmL_bond[ix1] = dmL[ix1]
dmL_bond[ix2] = dmL[ix2]

dm_bond = S12i @ dmL_bond @ S12i

np.save(molfile+'.dm_bond.npy', dm_bond)
