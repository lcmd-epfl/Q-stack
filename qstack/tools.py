import numpy as np

def _pyscf2gpr_idx(mol):
    """Given a molecule returns a list of reordered indices to tranform pyscf AO ordering into SA-GPR.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.
    """

    idx = np.arange(mol.nao_nr(), dtype=int)

    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]

        numbs = np.zeros(max_l+1, dtype=int)
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
    """Given a molecule returns a list of reordered indices to tranform SA-GPR AO ordering into pyscf.

    Args:
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        A numpy ndarray of re-arranged indices.
    """

    idx = np.arange(mol.nao_nr(), dtype=int)

    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        max_l = mol._basis[q][-1][0]

        numbs = np.zeros(max_l+1, dtype=int)
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


def pyscf2gpr(mol, vector):
    """Reorder p-orbitals from +1,-1,0 (pyscf convention) to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        vector (numpy ndarray): Vector or a matrix of atomic orbitals ordered in pyscf convention.

    Returns:
        A numpy ndarray with the reordered vector or matrix.
    """

    idx = _pyscf2gpr_idx(mol)
    dim = vector.ndim

    if dim == 1:
        return vector[idx]
    elif dim == 2:
        return vector[idx].T[idx]
    else:
        errstr = 'Dim = '+ str(dim)+' (should be 1 or 2)'
        raise Exception(errstr)

def gpr2pyscf(mol, vector):
    """Reorder p-orbitals from -1,0,+1 (SA-GPR convention) to +1,-1,0 (pyscf convention).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        vector (numpy ndarray): Vector or a matrix of atomic orbitals ordered in pyscf convention.

    Returns:
        A numpy ndarray with the reordered vector or matrix.
    """

    idx = _gpr2pyscf_idx(mol)
    dim = vector.ndim

    if dim == 1:
        return vector[idx]
    elif dim == 2:
        return vector[idx].T[idx]
    else:
        errstr = 'Dim = '+ str(dim)+' (should be 1 or 2)'
        raise Exception(errstr)


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


import resource
import time

def unix_time_decorator(func):
# thanks to https://gist.github.com/turicas/5278558
  def wrapper(*args, **kwargs):
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    ret = func(*args, **kwargs)
    end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
    print(func.__name__, ':  real: %.4f  user: %.4f  sys: %.4f'%
          (end_time - start_time,
           end_resources.ru_utime - start_resources.ru_utime,
           end_resources.ru_stime - start_resources.ru_stime))
    return ret
  return wrapper


def correct_num_threads():
    import os
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["MKL_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
