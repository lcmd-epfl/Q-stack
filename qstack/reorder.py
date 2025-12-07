"""Functions for reordering atomic orbitals between different conventions.

As the base convention, we use the "SA-GPR" ordering:
    m=-l, -l+1, ..., -1, 0, +1, ..., l-1, l

In PySCF, it is the same except for p-orbitals which are ordered as:
    if l==1:  m=+1, -1, 0                       (i.e., x,y,z)

In Orca, it is:
    m=0, +1, -1, +2, -2, ..., +l, -l
Additionally, Orca uses a different sign convention for |m|>=3.

In Gaussian, it is:
    if l==1:  m=+1, -1, 0                       (like in PySCF)
    if l>1:   m=0, +1, -1, +2, -2, ..., +l, -l  (like in Orca)

In Turbomole, it is:
    if l==1:  m=+1, -1, 0                       (like in PySCF)
    if l>1:   m=0, +1, -1, -2, +2, ..., -(-1)**(l%2)*l, (-1)**(l%2)*l
"""

import functools
import numpy as np
from .tools import slice_generator
from .compound import basis_flatten
from .constants import XYZ


_TURBOMOLE_MAX_L = 4


class MagneticOrder:
    """Class to handle ordering conventions of atomic orbitals.

    Args:
        l_checker (callable): Function that takes angular momentum l and returns
            True if the convention defines an order for that l.
        m_generator (callable or dict): Function that takes angular momentum l
            and returns the list of magnetic quantum numbers in the convention's order,
            or a dict mapping l to the corresponding list.
    """
    def __init__(self, l_checker=None, m_generator=None):
        self.check_l = l_checker
        self.m_generator = m_generator
        self.cache = {}

    def __contains__(self, l):
        return self.check_l(l)

    def __getitem__(self, l):
        if l not in self:
            return None
        elif l in self.cache:
            return self.cache[l]
        else:
            if isinstance(self.m_generator, dict):
                m_list = self.m_generator[l]
            else:
                m_list = self.m_generator(l)
            self.cache[l] = _order_from_m(l, m_list)
            return self.cache[l]


_orca_get_m  = lambda l: functools.reduce(lambda res, l1: [*res, l1, -l1], range(1, l+1), [0])
_turbo_get_m = lambda l: functools.reduce(lambda res, l1: [*res, l1, -l1] if l1%2 else [*res, -l1, l1], range(1, l+1), [0])

_conventions = {
        'gpr': MagneticOrder(
            l_checker=lambda _: False,
            ),
        'orca': MagneticOrder(
            l_checker=lambda l: l>0,
            m_generator=_orca_get_m,
            ),
        'pyscf': MagneticOrder(
            l_checker=lambda l: l==1,
            m_generator={1: XYZ},
            ),
        'gaussian': MagneticOrder(
            l_checker=lambda l: l>0,
            m_generator=lambda l: XYZ if l==1 else _orca_get_m(l),
            ),
        'turbomole': MagneticOrder(
            l_checker=lambda l: l>0,
            m_generator=lambda l: XYZ if l==1 else _turbo_get_m(l),
            ),
        }


def _order_from_m(l, current_m):
    """Get the indices to reorder atomic orbitals of angular momentum l to GPR order.

    Args:
        l (int): Angular momentum quantum number.
        current_m (iterable): Current order of magnetic quantum numbers.

    Returns:
        numpy.ndarray: Indices to reorder from order b to GPR order.

    Raises:
        ValueError: If b does not contain the correct m values for given l.
    """
    gpr_m = np.arange(-l, l+1)
    if any(sorted(current_m)!=gpr_m):
        raise ValueError("Wrong m values for given l")
    return np.equal.outer(gpr_m, current_m).argmax(axis=1)


def _conv2gpr_idx(nao, l_slices, convention):
    """Create the indices to reorder atomic orbitals from a given convention to SA-GPR.

    Args:
        nao (int): Number of atomic orbitals.
        l_slices (iterator): Iterator that yeilds (l: int, s: slice) per shell, where
            l is angular momentum quantum number and s is the corresponding slice of size 2*l+1.
        convention (str): Ordering convention.

    Returns:
        numpy ndarray: Re-arranged indices array.
    """
    order = _conventions[convention]
    idx = np.arange(nao)
    for l, s in l_slices:
        if l in order:
            idx[s] = idx[s][order[l]]
    return idx


def _get_idx(l, m, shell_start, convention):
    """Get the indices and sign multipliers to convert from a given convention to SA-GPR.

    Args:
        l (numpy.ndarray): Array of magnetic quantum numbers per atomic orbital.
        m (numpy.ndarray): Array of magnetic quantum numbers per atomic orbital.
        shell_start (numpy.ndarray): Starting AO indices for each shell (2*l+1 block).
        convention (str): Ordering convention.

    Returns:
        tuple: (idx (numpy.ndarray), signs (numpy.ndarray))
            idx: Indices to reorder from given convention to SA-GPR.
            signs: Sign multipliers to convert from given convention to SA-GPR.

    Raises:
        NotImplementedError: If the specified convention is not implemented or if l>4 for Turbomole convention.
    """
    convention = convention.lower()
    l_shells = l[shell_start]
    l_slices = slice_generator(l_shells, inc=lambda l: 2*l+1)

    if convention not in _conventions:
        errstr = f'Conversion to/from the {convention} convention is not implemented'
        raise NotImplementedError(errstr)

    idx = _conv2gpr_idx(len(l), l_slices, convention)

    signs = np.ones_like(idx)
    if convention == 'orca':
        signs[np.where(np.abs(m)>=3)] = -1  # in pyscf order
    elif convention == 'turbomole':
        """
        To get this, use `infsao` command of Turbomole's `define` program.
        It will print AO order and equations for each spherical harmonic.
        Check if the phase convention is the same we use
        (https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics).
        """
        if max(l)>_TURBOMOLE_MAX_L:
            raise NotImplementedError(f"Phase convention differences orbitals with l>{_TURBOMOLE_MAX_L} are not implemented yet. You can contribute!")
        signs[(l==3) & (m==-3)] = -1  # in pyscf order
        signs[(l==4) & (m==-3)] = -1  # in pyscf order
        signs[(l==4) & (m== 2)] = -1  # in pyscf order
    signs[idx] = np.copy(signs)   # in convention order. copy for numpy < 2

    return idx, signs


def reorder_ao(mol, vector, src='pyscf', dest='gpr'):
    """Reorder the atomic orbitals from one convention to another.

    For example, src=pyscf dest=gpr reorders p-orbitals from +1,-1,0 (pyscf convention)
    to -1,0,+1 (SA-GPR convention).

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        vector (numpy.ndarray): Vector (nao,) or matrix (mol.nao,mol.nao) to reorder.
            If None, returns the indices to reorder and sign multipliers for an 1D vector
            to use as `x = x[idx]*sign`.
        src (str): Current convention. Defaults to 'pyscf'.
        dest (str): Convention to convert to (available: 'pyscf', 'gpr', 'orca', 'gaussian', 'turbomole'). Defaults to 'gpr'.

    Returns:
        numpy.ndarray: Reordered vector or matrix, or tuple of (idx (numpy.ndarray), sign (numpy.ndarray)) if vector is None.

    Raises:
        ValueError: If vector dimension is not 1 or 2.
    """
    if vector is not None and vector.ndim not in (1,2):
        errstr = f'Dim = {vector.ndim} (should be 1 or 2)'
        raise ValueError(errstr)

    if src.lower() == dest.lower():
        if vector is None:
            return np.arange(mol.nao), np.ones(mol.nao)
        else:
            return vector

    (_, l, m), shell_start = basis_flatten(mol, return_both=False, return_shells=True)

    idx_src, sign_src  = _get_idx(l, m, shell_start, src)
    idx_dest, sign_dest = _get_idx(l, m, shell_start, dest)

    if vector is None:
        idx = np.arange(mol.nao)
        idx[idx_dest] = idx[idx_src]
        sign = np.ones_like(idx)
        sign[idx_dest] = (sign_src*sign)[idx_src] * sign_dest[idx_dest]
        return idx, sign

    if vector.ndim == 2:
        sign_src  = np.einsum('i,j->ij', sign_src, sign_src)
        sign_dest = np.einsum('i,j->ij', sign_dest, sign_dest)
        idx_dest = np.ix_(idx_dest,idx_dest)
        idx_src  = np.ix_(idx_src,idx_src)

    newvector = np.zeros_like(vector)
    newvector[idx_dest] = (sign_src*vector)[idx_src] * sign_dest[idx_dest]
    return newvector
