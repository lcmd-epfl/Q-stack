#!/usr/bin/env python3
import sys, os
import numpy as np
import pyscf_ext

basis2 = 'ccpvdz jkfit'

geom_directory = sys.argv[1]
mol2_filename = sys.argv[2]

atom1 = 0
atom2 = 0

mol1_filenames  = sorted(filter(lambda x: not x.endswith('.npy'), os.listdir(geom_directory)))

N = len(mol1_filenames)


mol2 = pyscf_ext.readmol(mol2_filename, basis2)
id2  = mol2.aoslice_nr_by_atom()[atom2][2:]
c2   = np.load(mol2_filename+'.c.npy')[id2[0]:id2[1]]
S    = mol2.intor('int1e_ovlp')[id2[0]:id2[1],id2[0]:id2[1]]
Sc2  = S @ c2

mol1 = pyscf_ext.readmol(geom_directory+'/'+mol1_filenames[0], basis2)
id1  = mol1.aoslice_nr_by_atom()[atom1][2:]

S_sum = 0.0
c1_sum = np.zeros_like(c2)

S_full  = 0.0
S2_full = 0.0
for mol1_filename in mol1_filenames:
    c1 = np.load(geom_directory+'/'+mol1_filename+'.c.npy')[id1[0]:id1[1]]
    ds = c1 @ Sc2
    S_full += ds
    S2_full += ds*ds

S_full  /= N
S2_full /= N

print(N, S_full, S2_full)

