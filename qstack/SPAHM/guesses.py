import sys
import numpy
import scipy
import pyscf
import pyscf.dft
from LB2020guess import LB2020guess

def hcore(mol, *_):
  h  = mol.intor_symmetric('int1e_kin')
  h += mol.intor_symmetric('int1e_nuc')
  return h

def GWH(mol, *_):
  h = hcore(mol)
  S = mol.intor_symmetric('int1e_ovlp')
  K = 1.75 # See J. Chem. Phys. 1952, 20, 837
  h_gwh = numpy.zeros_like(h)
  for i in range(h.shape[0]):
    for j in range(h.shape[1]):
      if i != j:
        h_gwh[i,j] = 0.5 * K * (h[i,i] + h[j,j]) * S[i,j]
      else:
        h_gwh[i,j] = h[i,i]
  return h_gwh

def SAD(mol, func):
  hc = hcore(mol)
  dm =  pyscf.scf.hf.init_guess_by_atom(mol)
  mf = pyscf.dft.RKS(mol)
  mf.xc = func
  vhf = mf.get_veff(dm=dm)
  fock = hc + vhf
  return fock

def SAP(mol, *_):
  mf = pyscf.dft.RKS(mol)
  vsap = mf.get_vsap()
  t = mol.intor_symmetric('int1e_kin')
  fock = t + vsap
  return fock

def LB(mol, *_):
  return LB2020guess(parameters='HF').Heff(mol)

def LB_HFS(mol, *_):
  return LB2020guess(parameters='HFS').Heff(mol)

def solveF(mol, fock):
  s1e = mol.intor_symmetric('int1e_ovlp')
  return scipy.linalg.eigh(fock, s1e)

def get_guess(arg):
  arg = arg.lower()
  guesses = {'core':hcore, 'sad':SAD, 'sap':SAP, 'gwh':GWH, 'lb':LB, 'huckel':None, 'lb-hfs':LB_HFS}
  if arg not in guesses.keys():
    print('Unknown guess. Available guesses:', list(guesses.keys()), file=sys.stderr);
    exit(1)
  return guesses[arg]

def get_occ(e, nelec, spin):
  if spin==None:
    nocc = nelec[0]
    return e[:nocc]
  else:
    nocc = nelec
    e1 = numpy.zeros((2, nocc[0]))
    e1[0,:nocc[0]] = e[:nocc[0]]
    e1[1,:nocc[1]] = e[:nocc[1]]
    return e1

