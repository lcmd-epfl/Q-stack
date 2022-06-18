#!/usr/bin/env python3

import os,sys
import numpy as np
import qstack
from pyscf.data.elements import ELEMENTS

geom_directory = sys.argv[1]
geom_directory = geom_directory+'/'
mol_filenames  = sorted(os.listdir(geom_directory))

tostr = False if (len(sys.argv)>2 and sys.argv[2]=='i') else True

q = []
for i,f in enumerate(mol_filenames):
  print(f)
  mol = qstack.compound.xyz_to_mol(geom_directory+f, 'sto3g')
  q.extend(mol.atom_charges())
if tostr:
  q = list(map(lambda x: ELEMENTS[x], q))
np.savetxt('q.dat', np.array(q), fmt="%s" if tostr else "%d")
