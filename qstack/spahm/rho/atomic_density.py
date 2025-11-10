"""Atomic density computation."""

import numpy as np
from qstack import compound, fields
from . import lowdin


def fit(mol, dm, aux_basis, short=False, w_slicing=True, only_i=None):
    """Create atomic density representations using Löwdin partitioning and density fitting.

    Decomposes the molecular density matrix into atomic contributions using Löwdin
    orthogonalization, then fits each atomic density onto auxiliary basis set.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): 2D density matrix in AO basis.
        aux_basis (str or dict): Auxiliary basis set for density fitting.
        short (bool): If True, returns only diagonal blocks (atom-centered coefficients).
            Defaults to False.
        w_slicing (bool): If True, uses block-diagonal Coulomb matrix (faster).
            Defaults to True.
        only_i (list or None): List of atom indices to compute. If None, computes all atoms.
            Defaults to None.

    Returns:
        list or numpy ndarray: Density fitting coefficients for each atom.
        - If short=False: list of 1D arrays (full aux basis per atom)
        - If short=True: 1D array (concatenated atom-centered coefficients only)
    """
    L = lowdin.Lowdin_split(mol, dm)

    if only_i is not None and len(only_i) > 0:
        dm_slices = mol.aoslice_nr_by_atom()[only_i,2:]
    else:
        dm_slices = mol.aoslice_nr_by_atom()[:,2:]

    auxmol = compound.make_auxmol(mol, aux_basis)
    eri2c, eri3c = fields.decomposition.get_integrals(mol, auxmol)[1:]

    if w_slicing:
        fit_slices = auxmol.aoslice_nr_by_atom()[:,2:]
        J = np.zeros_like(eri2c)
        for s0,s1 in fit_slices:
            J[s0:s1, s0:s1] = eri2c[s0:s1, s0:s1]
    else:
        fit_slices = None
        J = eri2c

    a_dfs = []
    for (start, stop) in dm_slices:
        a_dm1 = np.zeros_like(L.dmL)
        a_dm1[start:stop,:] += L.dmL[start:stop,:]*0.5
        a_dm1[:,start:stop] += L.dmL[:,start:stop]*0.5
        a_dm0 = L.S12i @ a_dm1 @ L.S12i

        c_a = fields.decomposition.get_coeff(a_dm0, J, eri3c, slices=fit_slices)
        a_dfs.append(c_a)

    if short:
        if only_i is not None and len(only_i) > 0:
            aoslice_by_atom = auxmol.aoslice_by_atom()[only_i,2:]
        else:
            aoslice_by_atom = auxmol.aoslice_by_atom()[:,2:]
        return [c[i0:i1] for (i0, i1), c in zip(aoslice_by_atom, a_dfs, strict=True)]

    return a_dfs
