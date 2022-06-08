import operator
import numpy as np
from pyscf import gto
from qstack import fields

import modules.repr as repre
from modules import lowdin
from modules.Dmatrix import Dmatrix_for_z, c_split, rotate_c

def make_bname(q0,q1):
    return operator.concat(*sorted((q0, q1)))

def get_basis_info(qqs, mybasis, only_m0, printlevel):
  idx = {}
  M   = {}
  for qq in qqs:
    if printlevel>1: print(qq)
    S, ao, _  = repre.get_S('No', mybasis[qq])
    if not only_m0:
      idx[qq] = repre.store_pair_indices_z(ao)
    else:
      idx[qq] = repre.store_pair_indices_z_only0(ao)
    M[qq]     = repre.metrix_matrix_z('No', idx[qq], ao, S)

  return idx, M

def read_df_basis(bnames, bpath):
    mybasis = {}
    for bname in bnames:
        if bname in mybasis: continue
        with open(bpath+'/'+bname+'.bas', 'r') as f:
            mybasis[bname] = eval(f.read())
    return mybasis


def get_element_pairs(elements):
    qqs0  = []
    qqs4q = {}
    for q1 in elements:
        qqs4q[q1] = []
        for q2 in elements:
            qq = make_bname(q1,q2)
            qqs4q[q1].append(qq)
            qqs0.append(qq)
        qqs4q[q1].sort()
    qqs0 = sorted(set(qqs0))
    qqs = {}
    for q in elements:
        qqs[q] = qqs0
    return qqs, qqs4q

def read_basis_wrapper(mols, bpath, only_m0, printlevel):
    elements  = sorted(list(set([q for mol in mols for q in mol.elements])))
    qqs,qqs4q = get_element_pairs(elements)
    qqs0      = qqs[list(qqs.keys())[0]]
    mybasis   = read_df_basis(qqs0, bpath)
    idx, M    = get_basis_info(qqs0, mybasis, only_m0, printlevel)
    return elements, mybasis, qqs, qqs4q, idx, M




def bonds_dict_init(qqs, M):
    N = 0
    mybonds = {}
    for qq in qqs:
        n = len(M[qq])
        mybonds[qq] = np.zeros(n)
        N += n
    return mybonds, N


def fit_dm(dm, mol, mybasis, ri0, ri1):
    rm = (ri0+ri1)*0.5
    atom = "No  % f % f % f" % (rm[0], rm[1], rm[2])
    auxmol = gto.M(atom=atom, basis=mybasis)

    e2c, e3c = fields.decomposition.get_integrals(mol, auxmol)[1:]
    c = fields.decomposition.get_coeff(dm, e2c, e3c)
    cs = c_split(auxmol, c)
    return cs

def vec_from_cs(z, cs, lmax, idx):
    D = Dmatrix_for_z(z, lmax)
    c_new = rotate_c(D, cs)
    v = repre.vectorize_c('No', idx, c_new)
    return v



def repr_for_mol(mol, dm, qqs, M, mybasis, idx, maxlen, cutoff):

  L = lowdin.Lowdin_split(mol, dm)
  q = [mol.atom_symbol(i) for i in range(mol.natm)]
  r = mol.atom_coords(unit='ANG')

  mybonds = [bonds_dict_init(qqs[q0], M) for q0 in q]

  for i0 in range(mol.natm):
    for i1 in range(i0):
      q0, q1 = q[i0], q[i1]
      r0, r1 = r[i0], r[i1]
      z = r1-r0
      if np.linalg.norm(z)>cutoff:
        #1e-12: 6 A
        #1e-10: 5 A
        #1e-8 : 4 A
        #1e-6 : 3 A
        continue

      dm1   = L.get_bond(i0,i1)
      bname = make_bname(q0,q1)
      cs    = fit_dm(dm1, mol, mybasis[bname], r0, r1)
      lmax  = max([ c[0] for c in cs])
      v0    = vec_from_cs(+z, cs, lmax, idx[bname])
      v1    = vec_from_cs(-z, cs, lmax, idx[bname])
      mybonds[i0][0][bname] += v0
      mybonds[i1][0][bname] += v1

  vec = [None]*mol.natm
  for i0 in range(mol.natm):
      vec[i0] = np.hstack([ M[qq] @ mybonds[i0][0][qq] for qq in qqs[q[i0]] ])
      vec[i0] = np.pad(vec[i0], (0, maxlen-len(vec[i0])), 'constant')
  return np.array(vec)
