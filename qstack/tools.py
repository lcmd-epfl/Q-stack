import os
import time
import resource
import numpy as np


class FrozenKeysDict:
    def __init__(self, keys=None, dictionary=None):
        if keys is not None:
            self._dictionary = dict.fromkeys(keys)
            if dictionary is not None:
                for key in dictionary:
                    self.__setitem__(key, dictionary[key])
        elif dictionary is not None:
            self._dictionary = dict(dictionary.items())  #  copy
        else:
            raise ValueError('Either keys or dictionary must be provided')

    def keys(self):
        return self._dictionary.keys()

    def __setitem__(self, key, item):
        if key not in self._dictionary:
            raise KeyError(f'The key {key} is not defined by {self.__class__.__name__}')
        self._dictionary[key] = item

    def __contains__(self, key):
        return key in self._dictionary

    def __getitem__(self, key):
        return self._dictionary[key]

    def __str__(self):
        return str(self._dictionary)

    def __repr__(self):
        return repr(self._dictionary)


def _orca2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform orca AO ordering into SA-GPR.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.
    """
    #def _M1(n):
    #    return (n+1)//2 if n%2 else -((n+1)//2)
    idx = np.arange(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            for _n in range(nf):
                #for m in range(-l, l+1):
                #    m1 = _M1(m+l)
                #    idx[(i+(m1-m))] = i
                #    i+=1
                I = np.s_[i:i+msize]
                idx[I] = np.concatenate((idx[I][::-2], idx[I][1::2]))
                i += msize
    return idx


def _orca2gpr_sign(mol):
    """Given a molecule returns a list of multipliers needed to tranform from orca AO.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of +1/-1 multipliers
    """
    signs = np.ones(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            if l<3:
                i += msize*nf
            else:
                for _n in range(nf):
                    signs[i+5:i+msize] = -1  # |m| >= 3
                    i+= msize
    return signs


def _pyscf2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.
    """

    idx = np.arange(mol.nao, dtype=int)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        for gto in mol._basis[q]:
            l = gto[0]
            msize = 2*l+1
            nf = max([len(prim)-1 for prim in gto[1:]])
            if l==1:
                for _n in range(nf):
                    idx[i:i+3] = [i+1,i+2,i]
                    i += 3
            else:
                i += msize * nf
    return idx


def reorder_ao(mol, vector, src='pyscf', dest='gpr'):
    """Reorder the atomic orbitals from one convention to another.
    For example, src=pyscf dest=gpr reorders p-orbitals from +1,-1,0 (pyscf convention) to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        vector (numpy ndarray): vector or matrix
        src (string): current convention
        dest (string): convention to convert to (available: 'pyscf', 'gpr', ...

    Returns:
        A numpy ndarray with the reordered vector or matrix.
    """

    def get_idx(mol, convention):
        convention = convention.lower()
        if convention == 'gpr':
            return np.arange(mol.nao)
        elif convention == 'pyscf':
            return _pyscf2gpr_idx(mol)
        elif convention == 'orca':
            return _orca2gpr_idx(mol)
        else:
            errstr = f'Conversion to/from the {convention} convention is not implemented'
            raise NotImplementedError(errstr)

    def get_sign(mol, convention):
        convention = convention.lower()
        if convention in ['gpr', 'pyscf']:
            return np.ones(mol.nao, dtype=int)
        elif convention == 'orca':
            return _orca2gpr_sign(mol)

    idx_src  = get_idx(mol, src)
    idx_dest = get_idx(mol, dest)
    sign_src  = get_sign(mol, src)
    sign_dest = get_sign(mol, dest)

    if vector.ndim == 2:
        sign_src  = np.einsum('i,j->ij', sign_src, sign_src)
        sign_dest = np.einsum('i,j->ij', sign_dest, sign_dest)
        idx_dest = np.ix_(idx_dest,idx_dest)
        idx_src  = np.ix_(idx_src,idx_src)
    elif vector.ndim!=1:
        errstr = f'Dim = {vector.ndim} (should be 1 or 2)'
        raise ValueError(errstr)

    newvector = np.zeros_like(vector)
    newvector[idx_dest] = (sign_src*vector)[idx_src]
    newvector *= sign_dest

    return newvector


def _Rz(a):
    """Computes the rotation matrix around absolute z-axis.

    Args:
        a (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = np.cos(a)
    A[0,1] = -np.sin(a)
    A[0,2] = 0
    A[1,0] = np.sin(a)
    A[1,1] = np.cos(a)
    A[1,2] = 0
    A[2,0] = 0
    A[2,1] = 0
    A[2,2] = 1
    return A


def _Ry(b):
    """Computes the rotation matrix around absolute y-axis.

    Args:
        b (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = np.cos(b)
    A[0,1] = 0
    A[0,2] = np.sin(b)
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = 0
    A[2,0] = -np.sin(b)
    A[2,1] = 0
    A[2,2] = np.cos(b)
    return A

def _Rx(g):
    """Computes the rotation matrix around absolute x-axis.

    Args:
        g (float): Rotation angle.

    Returns:
        A 2D numpy ndarray containing the rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = 0
    A[1,0] = 0
    A[1,1] = np.cos(g)
    A[1,2] = -np.sin(g)
    A[2,0] = 0
    A[2,1] = np.sin(g)
    A[2,2] = np.cos(g)
    return A


def rotate_euler(a, b, g, rad=False):
    """Computes the rotation matrix given Euler angles.

    Args:
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool) : Wheter the angles are in radians or not.

    Returns:
        A 2D numpy ndarray with the rotation matrix.
    """

    if not rad:
        a = a * np.pi / 180
        b = b * np.pi / 180
        g = g * np.pi / 180

    A = _Rz(a)
    B = _Ry(b)
    G = _Rx(g)

    return A@B@G


def unix_time_decorator(func):
# thanks to https://gist.github.com/turicas/5278558
  def wrapper(*args, **kwargs):
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    ret = func(*args, **kwargs)
    end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
    real = end_time - start_time
    user = end_resources.ru_utime - start_resources.ru_utime
    syst = end_resources.ru_stime - start_resources.ru_stime
    print(f'{func.__name__} :  real: {real:.4f}  user: {user:.4f}  sys: {syst:.4f}')
    return ret
  return wrapper

def unix_time_decorator_with_tvalues(func):
# thanks to https://gist.github.com/turicas/5278558
  def wrapper(*args, **kwargs):
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    ret = func(*args, **kwargs)
    end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
    timing = {'real' : end_time - start_time,
              'user' : end_resources.ru_utime - start_resources.ru_utime,
              'sys' : end_resources.ru_stime - start_resources.ru_stime}
    return timing, ret
  return wrapper

def correct_num_threads():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["MKL_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
