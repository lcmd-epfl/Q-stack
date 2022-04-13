#!/usr/bin/env python3
import sys
import numpy as np
from pyscf import dft,df,tools
import pyscf_ext

basis  = 'ccpvdz'
basis2 = 'ccpvdz jkfit'

mol = pyscf_ext.readmol(sys.argv[1], basis)

mf = dft.RKS(mol)
mf.xc = 'pbe'

mf.run()

auxmol = df.make_auxmol(mol, basis2)
e2c = auxmol.intor('int2c2e_sph')
e3c = pyscf_ext.eri_pqi(mol, auxmol)
w   = np.einsum('pq,qpi->i', mf.make_rdm1(), e3c)
c   = np.linalg.solve(e2c, w)

np.save(sys.argv[1]+'.c.npy', c)
