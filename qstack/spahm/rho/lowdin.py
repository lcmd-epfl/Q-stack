import sys
import numpy as np
from pyscf import dft,df,tools

class Lowdin_split:

  def __init__(self, mol, dm):
    S = mol.intor_symmetric('int1e_ovlp')
    S12,S12i = self.sqrtm(S)
    self.S    = S
    self.S12  = S12
    self.S12i = S12i
    self.mol  = mol
    self.dm   = dm
    self.dmL  = S12 @ dm @ S12

  def sqrtm(self, m):
    e,b = np.linalg.eigh(m)
    e   = np.sqrt(e)
    sm  = b @ np.diag(e    ) @ b.T
    sm1 = b @ np.diag(1.0/e) @ b.T
    return (sm+sm.T)*0.5, (sm1+sm1.T)*0.5

  def get_bond(self, at1idx, at2idx):
    mo1idx = range(*self.mol.aoslice_nr_by_atom()[at1idx][2:])
    mo2idx = range(*self.mol.aoslice_nr_by_atom()[at2idx][2:])
    ix1 = np.ix_(mo1idx,mo2idx)
    ix2 = np.ix_(mo2idx,mo1idx)
    dmL_bond = np.zeros_like(self.dmL)
    dmL_bond[ix1] = self.dmL[ix1]
    dmL_bond[ix2] = self.dmL[ix2]
    return self.S12i @ dmL_bond @ self.S12i

