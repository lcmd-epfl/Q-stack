import numpy as np
from qstack import compound, fields
from . import lowdin


def fit(mol, dm, aux_basis, short=False, w_slicing=True, only_i=[]):

    L = lowdin.Lowdin_split(mol, dm)

    if len(only_i) != 0:
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
        J = eri2c

    a_dfs = []
    for (start, stop) in dm_slices:
        a_dm1 = np.zeros_like(L.dmL)
        a_dm1[start:stop,:] += L.dmL[start:stop,:]*0.5
        a_dm1[:,start:stop] += L.dmL[:,start:stop]*0.5
        a_dm0 = L.S12i @ a_dm1 @ L.S12i

        c_a = fields.decomposition.get_coeff(a_dm0, J, eri3c)
        a_dfs.append(c_a)

    if short:
        cc = []
        for i, c in zip(auxmol.aoslice_by_atom()[:,2:], a_dfs):
            cc.append(c[i[0]:i[1]])
        return np.hstack(cc)

    return a_dfs
