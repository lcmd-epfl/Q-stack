#!/usr/bin/env python3.6

import numpy as np
import pyscf_ext
from Dmatrix import Dmatrix_for_z, c_split, rotate_c

basis = 'ccpvdzjkfit'

for xyzfile in ['test_hf/a.xyz', 'test_hf/x.xyz', 'test_hf/y.xyz', 'test_hf/z.xyz']:

  mol = pyscf_ext.readmol(xyzfile, basis)

  c = np.load(xyzfile+'.c.npy')
  cs = c_split(mol,c)
  lmax = max([ c[0] for c in cs])

  r = mol.atom_coords()
  z = r[1]-r[0]

  D = Dmatrix_for_z(z, lmax)

  c_new = rotate_c(D, cs)

  np.save(xyzfile+'.c_new.npy', c_new)


