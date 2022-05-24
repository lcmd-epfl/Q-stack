#!/usr/bin/env python3
import sys
import numpy as np
from qstack import compound, fields

basis  = 'ccpvdz'
basis2 = 'ccpvdz jkfit'

mol       = compound.xyz_to_mol(sys.argv[1], basis)
dm        = fields.dm.get_converged_dm(mol, xc="pbe")
auxmol, c = fields.decomposition.decompose(mol, dm, basis2)
np.save(sys.argv[1]+'.c.npy', c)
