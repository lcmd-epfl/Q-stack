import numpy as np
import scipy
from pyscf import scf
from qstack import compound

def decompose(mol, dm, auxbasis):
    """Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        dm (2D numpy array): Density matrix.
        auxbasis (string / pyscf basis dictionary): Atom-centered basis to decompose on.

    Returns:
        A copy of the pyscf Mole object with the auxbasis basis in a pyscf Mole object, and a 1D numpy array containing the decomposition coefficients.
    """

    auxmol = compound.make_auxmol(mol, auxbasis)
    S, eri2c, eri3c = get_integrals(mol, auxmol)
    c = get_coeff(dm, eri2c, eri3c)
    return auxmol, c

def get_integrals(mol, auxmol):
    """Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        Three numpy ndarray containing: the overlap matrix, the 2-centers ERI matrix, and the 3-centers ERI matrix respectively.
    """

    # Get overlap integral in the auxiliary basis
    S = auxmol.intor('int1e_ovlp_sph')

    # Concatenate standard and auxiliary basis set into a pmol object
    pmol = mol + auxmol

    # Compute 2- and 3-centers ERI integrals using the concatenated mol object
    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

    return S, eri2c, eri3c

def get_self_repulsion(mol, dm):
    """Computes the Einstein summation of the Coulumb matrix and the density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Density matrix.

    Returns:
        A nummpy ndarray result of the Einstein summation of the J matrix and the Density matrix.
    """

    try:
        j, k = mol.get_jk()
    except:
        j, k = scf.hf.get_jk(mol, dm)
    return np.einsum('ij,ij', j, dm)

def decomposition_error(self_repulsion, c, eri2c):
    """Computes the decomposition error.

    .. todo::
        Write the complete docstring
    """
    return self_repulsion - c @ eri2c @ c

def get_coeff(dm, eri2c, eri3c, slices=None):
    """Computes the density expansion coefficients.

    Args:
        dm (numpy ndarray): Density matrix.
        eri2c (numpy ndarray): 2-centers ERI matrix.
        eri3c (numpy ndarray): 3-centers ERI matrix.
        slices (optional numpy ndarray): assume that eri2c is bloc-diagonal, by giving the boundaries of said blocks

    Returns:
        A numpy ndarray containing the expansion coefficients of the density onto the auxiliary basis.
    """

    # Compute the projection of the density onto auxiliary basis using a Coulomb metric
    projection = np.einsum('ijp,ij->p', eri3c, dm)

    # Solve Jc = projection to get the coefficients
    if slices is not None:
        if not (slices.ndim==2 and slices.shape[0]>0 and slices.shape[1]==2) or\
           not (slices[0,0] == 0 and slices[-1,1] == projection.shape[0]):
            raise RuntimeError(f"Wrong argument {slices=}")

        c = np.empty_like(projection)
        for s0,s1 in slices:
            c[s0:s1] = scipy.linalg.solve(eri2c[s0:s1,s0:s1], projection[s0:s1], assume_a='pos')
    else:
        c = scipy.linalg.solve(eri2c, projection, assume_a='pos')

    return c

def _get_inv_metric(mol, metric, v):
  """

  Args:
    mol (pyscf Mole): pyscf Mole object.
    metric (str): unit, overlap or coulomb.
    v (numpy ndarray): Number of electrons decomposed into a vector.
  """
  if isinstance(metric, str):
      metric = metric.lower()
      if metric in ['u', 'unit', '1']:
          return v
      elif metric in ['s', 'overlap', 'ovlp']:
          O = mol.intor('int1e_ovlp_sph')
      elif metric in ['j', 'coulomb']:
          O = mol.intor('int2c2e_sph')
  else:
      O = metric
  return scipy.linalg.solve(O, v, assume_a='pos')


def correct_N_atomic(mol, N, c0, metric='u'):
    """

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        N (int): Number of electrons. Defaults to None.
        c0 (1D numpy array): Decomposition coefficients.
        metric (str): .Defaults to 'u'.

    Returns:

    .. todo::
        Write the complete docstring.
    """

    Q   = number_of_electrons_deco_vec(mol, per_atom=True)
    N0  = c0 @ Q
    O1q = _get_inv_metric(mol, metric, Q)
    la  = scipy.linalg.solve(Q.T @ O1q, N-N0)
    c   = c0 + np.einsum('pq,q->p', O1q, la)
    return c


def correct_N(mol, c0, N=None, mode='Lagrange', metric='u'):
    """

    Args:
        mol (pyscf Mole): pyscf Mole objec used for the computation of the density matrix.
        c0 (1D numpy array): Decomposition coefficients.
        N (int): Number of electrons. Defaults to None.
        mode (str): Defaults to Lagrange.
        metric (str): Defaults to u.

    Returns:
        A numpy ndarray containing a set of expansion coefficients taking into account the correct total number of electrons.

    .. todo::
        Write the complete docstring.
    """

    mode = mode.lower()
    q = number_of_electrons_deco_vec(mol)
    N0 = c0 @ q

    if N is None:
        N = mol.nelectron

    if mode == 'scale':
        c = c0 * N/N0

    elif mode == 'lagrange' :
        O1q = _get_inv_metric(mol, metric, q)
        la  = (N - N0) / (q @ O1q)
        c   = c0 + la * O1q
    return c


def number_of_electrons_deco_vec(mol, per_atom=False):
    """

    .. todo::
        Write the complete docstring.
    """
    if per_atom:
        Q = np.zeros((mol.nao,mol.natm))
    else:
        Q = np.zeros(mol.nao)
    i = 0
    for iat in range(mol.natm):
        for bas_id in mol.atom_shell_ids(iat):
            l = mol.bas_angular(bas_id)
            n = mol.bas_nctr(bas_id)
            if l==0:
                w = mol.bas_ctr_coeff(bas_id)
                a = mol.bas_exp(bas_id)
                q = pow (2.0*np.pi/a, 0.75) @ w
                if per_atom:
                    Q[i:i+n,iat] = q
                else:
                    Q[i:i+n] = q
            i += (2*l+1)*n
    return Q

def number_of_electrons_deco(auxmol, c):
    """Computes the number of electrons of a molecule given a set of expansion coefficients and a Mole object.

    Args:
        auxmol (pyscf Mole): pyscf mol object holding molecular structure, composition and the auxiliary basis set.
        c (numpy ndarray): expansion coefficients of the density onto the auxiliary basis.

    Returns:
        The number of electrons as an integer value.
    """

    q = number_of_electrons_deco_vec(auxmol)
    return q @ c
