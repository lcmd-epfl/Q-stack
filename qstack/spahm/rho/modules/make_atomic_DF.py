import numpy as np
from scipy.linalg import sqrtm
from qstack import compound, fields

def get_a_DF(mol, dm, basis, aux_basis) :
    n_atm = mol.natm

    S = mol.intor_symmetric('int1e_ovlp')
    S_sqrt = sqrtm(S)
    S_inv = np.linalg.inv(S_sqrt)

    dm1 = S_sqrt @ dm @ S_sqrt

    dm_slices = mol.aoslice_nr_by_atom()
    a_dfs = []
    aux_mol = compound.make_auxmol(mol, aux_basis)
    fit_slices = aux_mol.aoslice_nr_by_atom()
    S_fit, eri2c, eri3c = fields.decomposition.get_integrals(mol, aux_mol)


    S_fit_a = np.zeros(S_fit.shape)
    for s in fit_slices :
        S_fit_a[s[2]:s[3], s[2]:s[3]] = S_fit[s[2]:s[3], s[2]:s[3]]
    S_fit_a_inv = np.linalg.inv(S_fit_a)
    mol_name = mol.atom.split('/')[-1].split('.')[0]
    print(f"\tFitting {n_atm} atomic density-matrices for : {mol_name} ...")
    for i in range (n_atm) :
        a_slice = dm_slices[i]
        a_slice_fit = fit_slices[i]
        a_dm1 = np.zeros(dm1.shape)
        start = a_slice[2]
        stop = a_slice[3]
        a_dm1[start:stop, start:stop] = dm1[start:stop, start:stop]
        a_dm1[start:stop, stop:] = 0.5 * dm1[start:stop, stop:]
        a_dm1[start:stop, :start] = 0.5 * dm1[start:stop, :start]
        a_dm1[:start, start:stop] = 0.5 * dm1[:start, start:stop]
        a_dm1[stop: , start:stop] = 0.5 * dm1[stop: , start:stop]
        a_dm0 = S_inv @ a_dm1 @ S_inv
        fit_start = a_slice_fit[2]
        fit_stop = a_slice_fit[3]
        df = fields.decomposition.get_coeff(a_dm0, eri2c, eri3c)
        # S_fit_inv = np.linalg.inv(S_fit)
        c_a = S_fit_a_inv @ df
        a_dfs.append(c_a)
#    a_dfs = np.array(a_dfs, dtype=object)
    print("\t... fitting completed !\n")
    # print(f"Fitted ({len(a_dfs)} coeffs.) using Lowdin model !")
    return a_dfs

