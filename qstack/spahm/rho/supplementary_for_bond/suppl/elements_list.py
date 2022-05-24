#!/usr/bin/env python3

import os,sys
import numpy as np
import pyscf_ext

geom_directory = sys.argv[1]
geom_directory = geom_directory+'/'
mol_filenames  = sorted(os.listdir(geom_directory))

q = []
for i,f in enumerate(mol_filenames):
  print(f)
  mol = pyscf_ext.readmol(geom_directory+f, 'sto3g')
  q.append(mol.atom_charges())
q = np.hstack(q)
np.savetxt('q.dat', q, fmt="%d")

