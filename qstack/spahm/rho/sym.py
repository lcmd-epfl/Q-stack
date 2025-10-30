import numpy as np
from qstack import compound
from qstack.mathutils.matrix import sqrtm


def idxl0(i, l, ao):
    # return the index of the basis function with the same L and N but M=0
    if l != 1:
        return i - ao['m'][i]+l
    else:
        return i + [0, 2, 1][ao['m'][i]]

def get_S(q, basis):
    mol = compound.make_atom(q, basis)
    S = mol.intor_symmetric('int1e_ovlp')

    l_per_bas, _n_per_bas, ao_start = compound.singleatom_basis_enumerator(mol._basis[q])

    ao = {'l': [], 'm': []}
    for l in l_per_bas:
        msize = 2*l+1
        ao['l'].extend([l]*msize)
        if l != 1:
            ao['m'].extend(np.arange(msize)-l)
        else:
            ao['m'].extend([1, -1, 0])  # x, y, z

    return S, ao, ao_start


def store_pair_indices(ao):
    idx = []
    for i, [li, mi] in enumerate(zip(ao['l'], ao['m'], strict=True)):
        for j, [lj, mj] in enumerate(zip(ao['l'], ao['m'], strict=True)):
            if (li!=lj) or (mi!=mj):
                continue
            idx.append([i, j])
    return idx


def store_pair_indices_short(ao, ao_start):
    idx = []
    for i in ao_start:
        for j in ao_start:
            li = ao['l'][i]
            lj = ao['l'][j]
            if li!=lj:
                continue
            idx.append([i, j])
    return idx


def metric_matrix(q, idx, ao, S):
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i,  j  = idx[p]
            i1, j1 = idx[p1]
            l  = ao['l'][i]
            l1 = ao['l'][i1]
            if(l!=l1):
                continue
            A[p1,p] = A[p,p1] = 1.0/(2*l+1) \
                                * S[idxl0(i, l, ao[q]), idxl0(i1, l, ao[q])] \
                                * S[idxl0(j, l, ao[q]), idxl0(j1, l, ao[q])]
    return sqrtm(A)


def metric_matrix_short(idx, ao, S):
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i, j  = idx[p]
            i1,j1 = idx[p1]
            l  = ao['l'][i]
            l1 = ao['l'][i1]
            if(l!=l1):
                continue
            A[p1,p] = A[p,p1] = S[i,i1] * S[j,j1] / (2*l+1)
    return sqrtm(A)


def vectorize_c(idx, c):
    v = np.zeros(len(idx))
    for p, (i,j) in enumerate(idx):
        v[p] = c[i]*c[j]
    return v


def vectorize_c_MR2021(idx_pair, ao, c):
    idx = sorted(set(np.array(idx_pair)[:,0]))
    v = np.zeros(len(idx))
    for p,i in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[i:i+msize]
    return v


def vectorize_c_short(idx, ao, c):
    v = np.zeros(len(idx))
    for p, [i,j] in enumerate(idx):
        l = ao['l'][i]
        msize = 2*l+1
        v[p] = c[i:i+msize] @ c[j:j+msize]
    return v


def store_pair_indices_z(ao):
    idx = []
    for i, mi in enumerate(ao['m']):
        for j, mj in enumerate(ao['m']):
            if abs(mi)!=abs(mj):
                continue
            idx.append([i,j])
    return idx


def store_pair_indices_z_only0(ao):
    idx = []
    for i, mi in enumerate(ao['m']):
        if mi!=0:
            continue
        for j, mj in enumerate(ao['m']):
            if mj!=0:
                continue
            idx.append([i,j])
    return idx


def metric_matrix_z(idx, ao, S):
    N = len(idx)
    A = np.zeros((N,N))
    for p in range(N):
        for p1 in range(p,N):
            i,i1 = idx[p]
            j,j1 = idx[p1]
            li  = ao['l'][i ]
            li1 = ao['l'][i1]
            lj  = ao['l'][j ]
            lj1 = ao['l'][j1]
            if (li != lj) or (li1 != lj1):
                continue

            mi  = ao['m'][i ]
            mi1 = ao['m'][i1]
            mj  = ao['m'][j ]
            mj1 = ao['m'][j1]

            A[p1,p] = A[p,p1] = ((mi==mj)*(mi1==mj1) + (mi==-mj)*(mi1==-mj1)*(mi!=0)) \
                                * S[idxl0(i,  li,  ao), idxl0(j,  li,  ao)] \
                                * S[idxl0(i1, li1, ao), idxl0(j1, li1, ao)]

    return sqrtm(A)
