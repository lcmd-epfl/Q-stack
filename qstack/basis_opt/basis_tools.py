import copy
import numpy as np
from pyscf import df,dft

def energy_mol(newbasis, moldata):
    """Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        numpy ndarray: Overlap matrix, 2-centers and 3-centers ERI matrices.

    """
    mol       = moldata['mol'      ]
    rho       = moldata['rho'      ]
    coords    = moldata['coords'   ]
    weights   = moldata['weights'  ]
    self      = moldata['self'     ]

    newmol = df.make_auxmol(mol, newbasis)
    ao = dft.numint.eval_ao(newmol, coords).T
    w = np.einsum('pi,i,i->p', ao,rho,weights)
    S = np.einsum('pi,qi,i->pq', ao,ao,weights)
    c = np.linalg.solve(S, w)
    E = self-c@w

    return E


def gradient_mol(nexp, bf_bounds, newbasis, moldata):

  mol       = moldata['mol'      ]
  rho       = moldata['rho'      ]
  coords    = moldata['coords'   ]
  weights   = moldata['weights'  ]
  self      = moldata['self'     ]
  idx       = moldata['idx'      ]
  centers   = moldata['centers'  ]
  distances = moldata['distances']

  newmol = df.make_auxmol(mol, newbasis)
  ao = dft.numint.eval_ao(newmol, coords).T

  w = np.einsum('pi,i,i->p', ao,rho,weights)
  
  dao_da_factor = np.ndarray((newmol.nao, len(rho)))  
  dw_da = np.zeros((nexp, newmol.nao))
  prev_ip = -1
  for p in range(newmol.nao):
    iat = centers[p]
    r2  = distances[iat]
    ip = idx[p]  # number of the 'bas' item within newbasis, not within the mol object
    if prev_ip != ip:
      s = newmol.atom_symbol(iat)
      bas = newbasis[s][ip - bf_bounds[s][0]]
    dao_da_factor[p,:] = 0.5*(bas[0]+1.5) / bas[1][0] -r2

    dw_da[ip,p] = np.einsum('i,i,i,i->', ao[p], rho, dao_da_factor[p], weights)

  S  = np.einsum('pi,qi,i->pq', ao,ao,weights)
  dS_da = np.zeros((nexp, newmol.nao, newmol.nao))

  for p in range(newmol.nao):
    for q in range(p,newmol.nao):
      ip = idx[p]
      iq = idx[q]
      ao_ao_w = np.einsum('i,i,i->i', ao[p],ao[q],weights)
      ip_p_q  = np.einsum('i,i->', ao_ao_w, dao_da_factor[p])
      iq_p_q  = np.einsum('i,i->', ao_ao_w, dao_da_factor[q])
      dS_da[ip,p,q] += ip_p_q
      dS_da[iq,p,q] += iq_p_q
      if p!=q:
        dS_da[ip,q,p] += ip_p_q
        dS_da[iq,q,p] += iq_p_q

  c = np.linalg.solve(S, w)
  part1 = np.einsum('p,ip->i',  c, dw_da)
  part2 = np.einsum('p,ipq,q->i',  c, dS_da, c)
  dE_da = 2.0*part1 - part2
  E = self - c@w
  return E, dE_da


def exp2basis(exponents, elements, basis):
  i=0
  newbasis = copy.deepcopy(basis)
  for q in elements:
    for j,b in enumerate(basis[q]):
      newbasis[q][j][1] = [exponents[i], 1]
      i+=1
  return newbasis


def cut_myelements(x, myelements, bf_bounds):
  x1 = []
  for q in myelements:
    bounds = bf_bounds[q]
    x1.append(x[bounds[0]:bounds[1]])
  x1 = np.concatenate(x1)
  return x1


def printbasis(basis, f):
  print('{', file=f)
  for q,b in basis.items():
    print('  "'+q+'": [', file=f)
    for i,gto in enumerate(b):
      if i > 0:
        print(',', file=f)
      print('   ', gto, file=f, end='')
    print('  ]', file=f)
  print('}', file=f)
