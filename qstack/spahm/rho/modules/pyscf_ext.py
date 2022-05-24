import warnings
import numpy
import pyscf
import pyscf.tools

#warnings.simplefilter('always', DeprecationWarning)

def makemol(xyz, basis, charge=0, spin=0):
  mol = pyscf.gto.Mole()
  mol.atom = xyz
  mol.charge = charge
  mol.spin = spin
  mol.basis = basis
  mol.build()
  return mol

def makeauxmol(mol, basis):
  warnings.warn("makeauxmol() is deprecated, use pyscf.df.make_auxmol() instead", DeprecationWarning)
  auxmol = pyscf.gto.Mole()
  auxmol.atom = mol.atom
  auxmol.charge = mol.charge
  auxmol.spin = mol.spin
  auxmol.basis = basis
  auxmol.build()
  return auxmol

def readmol(fname, basis, charge=0, spin=0, ignore=False):
  def _readxyz(fname):
    with open(fname, "r") as f:
      xyz = f.readlines()
    return "".join(xyz[2:])
  xyz = _readxyz(fname)
  if not ignore:
    mol = makemol(xyz, basis, charge, spin)
  else:
    try:
      mol = makemol(xyz, basis)
    except:
      mol = makemol(xyz, basis, -1)
  return mol

def _pyscf2gpr_idx(mol):

  idx = numpy.arange(mol.nao_nr(), dtype=int)

  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    max_l = mol._basis[q][-1][0]

    numbs = numpy.zeros(max_l+1, dtype=int)
    for gto in mol._basis[q]:
      l = gto[0]
      nf = max([len(prim)-1 for prim in gto[1:]])
      numbs[l] += nf

    i+=numbs[0]
    if(max_l<1):
      continue

    for n in range(numbs[1]):
      idx[i  ] = i+1
      idx[i+1] = i+2
      idx[i+2] = i
      i += 3

    for l in range(2, max_l+1):
      i += (2*l+1)*numbs[l]

  return idx

def _gpr2pyscf_idx(mol):

  idx = numpy.arange(mol.nao_nr(), dtype=int)

  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    max_l = mol._basis[q][-1][0]

    numbs = numpy.zeros(max_l+1, dtype=int)
    for gto in mol._basis[q]:
      l = gto[0]
      nf = max([len(prim)-1 for prim in gto[1:]])
      numbs[l] += nf

    i+=numbs[0]
    if(max_l<1):
      continue

    for n in range(numbs[1]):
      idx[i+1] = i
      idx[i+2] = i+1
      idx[i  ] = i+2
      i += 3

    for l in range(2, max_l+1):
      i += (2*l+1)*numbs[l]

  return idx


def _orca2gpr_idx(mol):
  def _M1(n):
    return (n+1)//2 if n%2 else -((n+1)//2)
  idx = numpy.arange(mol.nao_nr(), dtype=int)
  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    max_l = mol._basis[q][-1][0]
    numbs = numpy.zeros(max_l+1, dtype=int)
    for gto in mol._basis[q]:
      l = gto[0]
      nf = max([len(prim)-1 for prim in gto[1:]])
      numbs[l] += nf
    for l in range(max_l+1):
      for n in range(numbs[l]):
        for m in range(-l, l+1):
          m1 = _M1(m+l)
          idx[(i+(m1-m))] = i
          i+=1
  return idx

def _orca2gpr_sign(mol):
  def _M1(n):
    return (n+1)//2 if n%2 else -((n+1)//2)
  idx = numpy.arange(mol.nao_nr(), dtype=int)
  signs = numpy.array(mol.nao_nr() * [1])
  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    max_l = mol._basis[q][-1][0]
    numbs = numpy.zeros(max_l+1, dtype=int)
    for gto in mol._basis[q]:
      l = gto[0]
      nf = max([len(prim)-1 for prim in gto[1:]])
      numbs[l] += nf
    for l in range(max_l+1):
      for n in range(numbs[l]):
        for m in range(-l, l+1):
          if abs(m) >= 3:
            signs[i] = -1
          i+=1
  return signs

def pyscf2gpr(mol, vector):
  warnings.warn("pyscf2gpr() is deprecated, use reorder_ao instead", DeprecationWarning)
  return reorder_ao(mol, vector, 'pyscf2gpr')

def gpr2pyscf(mol, vector):
  warnings.warn("gpr2pyscf() is deprecated, use reorder_ao instead", DeprecationWarning)
  return reorder_ao(mol, vector, 'gpr2pyscf')

def reorder_ao(mol, vector, direction):
  if   direction == 'gpr2pyscf':
    idx = _gpr2pyscf_idx(mol)
    #idx2 = _pyscf2gpr_idx(mol)
    #idx3 = numpy.empty(len(idx2), dtype=int)
    #idx3[idx2] = numpy.arange(len(idx2))
    #idx = idx3
  elif direction == 'pyscf2gpr':
    idx = _pyscf2gpr_idx(mol)
  elif direction == 'orca2gpr':
    idx = _orca2gpr_idx(mol)
    signs = _orca2gpr_sign(mol)
  else:
    errstr = 'Unknown direction '+ direction
    raise Exception(errstr)
  dim = vector.ndim
  if dim == 1:
    newvector = vector[idx]
    if direction == 'orca2gpr':
      newvector = signs * newvector
  elif dim == 2:
    newvector = vector[idx].T[idx]
    if direction == 'orca2gpr':
      newvector = numpy.einsum('i,j->ij', signs, signs) * newvector
  else:
    errstr = 'Dim = '+ str(dim)+' (should be 1 or 2)'
    raise Exception(errstr)
  return newvector

def eri_pqi(mol, auxmol):
  pmol  = mol + auxmol
  eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
  return eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)


def number_of_electrons(rho, mol):
    nel = 0.0

    i = 0
    for iat in range(mol.natm):
        j = 0
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]
        numbs = [x[0] for x in mol._basis[q]]

        for n in range(numbs.count(0)):
            a, w = mol._basis[q][j][1]
            nel += rho[i] * w * pow (2.0*numpy.pi/a, 0.75) # norm = (2.0*a/np.pi)^3/4, integral = (pi/a)^3/2
            i += 1
            j += 1
        for l in range(1,max_l+1):
            n_l = numbs.count(l)
            i += n_l * (2*l+1)
            j += n_l

    return nel

def number_of_electrons_vec(mol):
    nel = []
    for iat in range(mol.natm):
        j = 0
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]
        numbs = [x[0] for x in mol._basis[q]]

        for n in range(numbs.count(0)):
            a, w = mol._basis[q][j][1]
            nel.append(w * pow (2.0*numpy.pi/a, 0.75))
            j += 1
        for l in range(1,max_l+1):
            n_l = numbs.count(l)
            nel.extend([0]*(2*l+1)*n_l)
            j += n_l
    return numpy.array(nel)


def matrix_from_tril(mat_tril):
  n = int((numpy.sqrt(1+8*len(mat_tril))-1)/2)
  ind = numpy.tril_indices(n)
  mat = numpy.zeros((n,n))
  mat[ind] = mat_tril
  mat = mat + mat.T - numpy.diag(numpy.diag(mat))
  return mat

def molden_orbital(mol, c, filename):
  with open(filename, 'w') as f:
    pyscf.tools.molden.header(mol, f, True)
    try:
      N = number_of_electrons(c,mol)
    except:
      N = 0.0
    pyscf.tools.molden.orbital_coeff(mol, f, numpy.array([c]).T, ene=[0.0], occ=[N], ignore_h=True)

