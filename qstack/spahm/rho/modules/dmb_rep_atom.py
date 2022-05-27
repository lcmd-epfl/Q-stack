import numpy as np
import pyscf
from qstack import fields
import modules.repr as repre
from modules import make_atomic_DF

def get_basis_info(atom_types, aux_basis_set):
    ao = {}
    idx = {}
    M = {}
    ao_len = {}
    for q in atom_types:
        S, ao[q], ao_start = repre.get_S(q, aux_basis_set)
        idx[q] = repre.store_pair_indices_short(ao[q], ao_start)
        M[q]   = repre.metrix_matrix_short(q, idx[q], ao[q], S)
        ao_len[q] = len(S)
    return ao, ao_len, idx, M


def get_model(arg):

    def df_pure(mol, dm, basis_set, aux_basis_set):
        return fields.decomposition.decompose(mol, dm, aux_basis_set)[1]
    def df_sad_diff(mol, dm, basis_set, aux_basis_set):
        mf = pyscf.scf.RHF(mol)
        dm_sad = mf.init_guess_by_atom(mol)
        dm = dm - dm_sad
        return fields.decomposition.decompose(mol, dm, aux_basis_set)[1]
    def df_lowdin_long(mol, dm, basis_set, aux_basis_set):
        return make_atomic_DF.get_a_DF(mol, dm, basis_set, aux_basis_set)
    def df_lowdin_short(mol, dm, basis_set, aux_basis_set):
        return make_atomic_DF.get_a_DF(mol, dm, basis_set, aux_basis_set, short=True)

    arg = arg.lower()
    models = {'pure'         : [df_pure,          coefficients_symmetrize_short],
             'sad-diff'      : [df_sad_diff,      coefficients_symmetrize_short],
             'lowdin-short'  : [df_lowdin_short,  coefficients_symmetrize_short],
             'lowdin-long'   : [df_lowdin_long,   coefficients_symmetrize_long ] }
    if arg not in models.keys():
        print('Unknown model. Available models:', list(models.keys()), file=sys.stderr)
        exit(1)
    return models[arg]



def coefficients_symmetrize_short(c, mol, idx, ao, ao_len, M, _):
    # short lowdin / everything else
    v = []
    i0 = 0
    for q in mol.elements :
        n = ao_len[q]
        v.append([q, M[q] @ repre.vectorize_c_short(q, idx[q], ao[q], c[i0:i0+n])])
        i0 += n
    return v

def coefficients_symmetrize_long(c_df, mol, idx, ao, ao_len, M, atom_types):
    # long lowdin
    vectors = []
    for c_a, e in zip(c_df, mol.elements) :
        v_atom = {}
        for q in atom_types:
            v_atom[q] = np.zeros(len(idx[q]))
        i0 = 0
        for q in mol.elements :
            n = ao_len[q]
            v_atom[q] += M[q] @ repre.vectorize_c_short(q, idx[q], ao[q], c_a[i0:i0+n])
            i0 += n
        v_a = np.hstack([v_atom[q] for q in atom_types])
        vectors.append([e, v_a])
    return vectors
