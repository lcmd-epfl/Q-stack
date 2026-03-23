"""Utility functions for basis set manipulation."""

import copy
import numpy as np
import scipy.linalg as spl
from pyscf import df, dft
import pyscf


def energy_mol(newbasis, moldata):
    """Compute loss function (fitting error) for one molecule.

    Args:
        newbasis (dict): Basis set.
        moldata (dict): Dictionary containing molecular data.

    Returns:
        float: Loss function value for the given basis.
    """
    mol     = moldata['mol'    ]
    rho     = moldata['rho'    ]
    coords  = moldata['coords' ]
    weights = moldata['weights']
    self    = moldata['self'   ]

    newmol = df.make_auxmol(mol, newbasis)
    ao = dft.numint.eval_ao(newmol, coords).T
    w = np.einsum('i,i,pi->p', rho, weights, ao)
    #S = np.einsum('i,pi,qi->pq', weights, ao, ao)
    #S = (ao*weights)@ao.T
    S = newmol.intor('int1e_ovlp')
    c = spl.solve(S, w, assume_a='pos')
    E = self-c@w

    return E


# import time
# def gradient_mol(nexp, newbasis, moldata):
#     """Compute loss function and gradient for one molecule with respect to basis exponents.

#     Args:
#         nexp (int): Number of exponents.
#         newbasis (dict): Basis set.
#         moldata (dict): Dictionary containing molecular data.

#     Returns:
#         tuple: A tuple containing:
#         - E (float): Loss function value.
#         - dE_da (numpy.ndarray): Gradient of loss function with respect to exponents.
#     """
#     start = time.perf_counter()

#     mol       = moldata['mol'      ]
#     rho       = moldata['rho'      ]
#     coords    = moldata['coords'   ]
#     weights   = moldata['weights'  ]
#     self      = moldata['self'     ]
#     idx       = moldata['idx'      ]
#     centers   = moldata['centers'  ]
#     distances = moldata['distances']

#     newmol = df.make_auxmol(mol, newbasis)
#     ao = dft.numint.eval_ao(newmol, coords).T

#     print(f'base eval done: {time.perf_counter()-start:f} s')
#     w = np.einsum('i,i,pi->p', rho, weights, ao)
#     dw_da = np.zeros((nexp, newmol.nao))
#     for p in range(newmol.nao):
#         iat = centers[p]
#         r2 = distances[iat]
#         dw_da[idx[p], p] = np.einsum('i,i,i,i->', ao[p], rho, r2, weights)
#     print(f'dw_da done: {time.perf_counter()-start:f} s')

#     #S = np.einsum('pi,qi,i->pq', ao, ao, weights)
#     S = newmol.intor('int1e_ovlp')
#     dS_da = np.zeros((nexp, newmol.nao, newmol.nao))

#     print(idx)
#     for p in range(newmol.nao):
#         for q in range(p, newmol.nao):
#             ip = idx[p]
#             iq = idx[q]
#             iatp = centers[p]
#             iatq = centers[q]
#             r2p = distances[iatp]
#             r2q = distances[iatq]
#             ao_ao_w = np.einsum('i,i,i->i', ao[p], ao[q], weights)
#             ip_p_q = np.einsum('i,i->', ao_ao_w, r2p)
#             iq_p_q = np.einsum('i,i->', ao_ao_w, r2q)
#             dS_da[ip, p, q] += ip_p_q
#             dS_da[iq, p, q] += iq_p_q
#             if p != q:
#                 dS_da[ip, q, p] += ip_p_q
#                 dS_da[iq, q, p] += iq_p_q
#     print(f'dS_da done: {time.perf_counter()-start:f} s')

#     c = spl.solve(S, w, assume_a='pos')
#     part1 = np.einsum('p,ip->i',  c, dw_da)
#     part2 = np.einsum('p,ipq,q->i',  c, dS_da, c)
#     dE_da = 2.0*part1 - part2
#     E = self - c@w
#     print(f'rest done: {time.perf_counter()-start:f} s')
#     return E, dE_da

import time
def gradient_mol(nexp, newbasis, moldata):
    """Compute loss function and gradient for one molecule with respect to basis exponents.

    Args:
        nexp (int): Number of exponents.
        newbasis (dict): Basis set.
        moldata (dict): Dictionary containing molecular data.

    Returns:
        tuple: A tuple containing:
        - E (float): Loss function value.
        - dE_da (numpy.ndarray): Gradient of loss function with respect to exponents.
    """
    #start = time.perf_counter()

    mol       = moldata['mol'      ]
    rho       = moldata['rho'      ]
    coords    = moldata['coords'   ]
    weights   = moldata['weights'  ]
    self      = moldata['self'     ]
    idx       = moldata['idx'      ]
    centers   = moldata['centers'  ]
    distances = moldata['distances']

    newmol = df.make_auxmol(mol, newbasis)
    ao = dft.numint.eval_ao(newmol, coords).T
    #S = np.einsum('i,pi,qi->pq', weights, ao, ao)
    #S = (ao*weights)@ao.T
    S = newmol.intor('int1e_ovlp')
    w = np.einsum('i,i,pi->p', rho, weights, ao)
    c = spl.solve(S, w, assume_a='pos')
    E = self - c@w
    del S, w
    #print(f'energy done: {time.perf_counter()-start:f} s')


    # ## the next block can also be expressed as:
    # ao_mapper = np.zeros(newmol.nao, nexp)
    # ao_mapper[ np.arange(newmol.nao), idx] = 1
    # dE_da = np.einsum('i,i,pi,pi,p,pc->c'
    #     rho, weights,
    #     distances[centers], ao,
    #     2*c,
    #     ao_mapper
    # )
    aoc_temp = ao * c[:, None]
    aoc_per_e = np.zeros((nexp, rho.shape[0]))
    for p,aoc in enumerate(aoc_temp):
        aoc *= distances[centers[p]]
        aoc_per_e[idx[p]] += aoc
    dE_da = 2* (aoc_per_e @ (rho*weights))
    del aoc_temp, aoc_per_e
    #print(f'dw part done: {time.perf_counter()-start:f} s')

    # ## check with original impl.
    # dw_da = np.zeros((nexp, newmol.nao))
    # for p in range(newmol.nao):
    #     iat = centers[p]
    #     r2 = distances[iat]
    #     dw_da[idx[p], p] = np.einsum('i,i,i,i->', ao[p], rho, r2, weights)
    # assert np.allclose(dE_da, 2*dw_da@c)
    
    
    # ## the next block can also be expressed as:
    # temp = np.einsum('i,pi,qi,p,q,pi->pq',
    #     weights, ao, ao, c, c,
    #     distances[centers],
    # )
    # temp += temp.T
    # temp.sum(axis=0)
    # dE_da -= temp @ ao_mapper  # dE_da -= c@dS_da@c
    # ## alternatively
    # dE_da -= np.einsum('i,pi,qi,p,q,pi,pc->c',
    #     weights, ao, ao, c, c,
    #     distances[centers],
    #     2*ao_mapper,
    # )

    aoc = c[:, None] * ao
    wao = aoc.sum(axis=0)
    wao *= weights

    mapped = np.zeros((nexp, rho.shape[0]))
    for p,aoc_slice in enumerate(aoc):
        mapped[idx[p]] += aoc_slice * distances[centers[p]]
    mapped *= wao

    # ## check with original impl.
    # dS_da = np.zeros((nexp, newmol.nao, newmol.nao))
    # for p in range(newmol.nao):
    #     for q in range(p, newmol.nao):
    #         ip = idx[p]
    #         iq = idx[q]
    #         iatp = centers[p]
    #         iatq = centers[q]
    #         r2p = distances[iatp]
    #         r2q = distances[iatq]
    #         ao_ao_w = np.einsum('i,i,i->i', ao[p], ao[q], weights)
    #         ip_p_q = np.einsum('i,i->', ao_ao_w, r2p)
    #         iq_p_q = np.einsum('i,i->', ao_ao_w, r2q)
    #         dS_da[ip, p, q] += ip_p_q
    #         dS_da[iq, p, q] += iq_p_q
    #         if p != q:
    #             dS_da[ip, q, p] += ip_p_q
    #             dS_da[iq, q, p] += iq_p_q
    # comparison = (dS_da@c)@c
    # assert np.allclose(2*mapped.sum(axis=1), comparison)


    
    dE_da -= 2*mapped.sum(axis=1)
    del mapped, wao, aoc
        
    #print(f'dS part done: {time.perf_counter()-start:f} s')
    return E, dE_da



def exp2basis(exponents, elements, basis):
    """Convert exponents array to basis set format.

    Args:
        exponents (numpy.ndarray): Array of basis function exponents.
        elements (list): List of elements for which change the basis.
        basis (dict): Template basis set definition.

    Returns:
        dict: New basis set with updated exponents.
    """
    i = 0
    newbasis = copy.deepcopy(basis)
    for q in elements:
        for j in range(len(basis[q])):
            newbasis[q][j][1] = [exponents[i], 1]
            i += 1
    return newbasis


def cut_myelements(x, myelements, bf_bounds):
    """Extract subset of array corresponding to specified elements.

    Args:
        x (numpy.ndarray): Input array.
        myelements (list): List of element symbols to extract.
        bf_bounds (dict): Dictionary mapping elements to their basis set bound indices.

    Returns:
        numpy.ndarray: Array containing x only for specified elements.
    """
    x1 = []
    for q in myelements:
        bounds = bf_bounds[q]
        x1.append(x[bounds[0]:bounds[1]])
    x1 = np.concatenate(x1)
    return x1


def printbasis(basis, f):
    """Print basis set in JSON-like format to file.

    Args:
        basis (dict): Basis set definition.
        f (file): File object to write to.
    """
    print('{', file=f)
    for i, (q, b) in enumerate(basis.items()):
        print('  "'+q+'": [', file=f)
        for i, gto in enumerate(b):
            if i > 0:
                print(',', file=f)
            print('   ', gto, file=f, end='')
        if i==len(basis)-1:
            print('\n  ]', file=f)
        else:
            print('\n  ],', file=f)
    print('}', file=f)


def basis_as_nwchem(f_out, basis_dict, name='custom basis set'):
    """Print basis set in NWchem format (the one pyscf reads from files)

    Prints a basis dictionnary into a provided file. The name of the basis, as indicated in the file itself, can be provided.

    Args:
        f_out (writable io.TextIOBase): the file to write the basis to
        basis_dict (pyscf-format basis dictionnary): the basis to write
        name (str): the optional name of the basis
    """

    sorted_atom_types = sorted(basis_dict.keys(), key=pyscf.data.elements.charge)
    angular_names = 'spdfgh'

    f_out.write(f"# {name:s}\n# (custom basis written by qstack)\n# supported elements: {', '.join(sorted_atom_types):s}\n\n")
    f_out.write("BASIS \"ao basis\" PRINT")
    for atom in sorted_atom_types:
        fakemol = pyscf.M(atom=atom, charge=pyscf.data.elements.charge(atom))
        nbas = fakemol.nbas
        l_count = max(fakemol.bas_angular(x) for x in range(nbas))+1

        prim_count=[0]*l_count
        shell_count=[0]*l_count
        for bas_i in range(nbas):
            l = fakemol.bas_angular(bas_i)
            prim_count[l] += fakemol.bas_nprim(bas_i)
            shell_count[l] += fakemol.bas_nctr(bas_i)
        prim_mark = [f"{n:d}{angular_names[l]:s}" for l,n in enumerate(prim_count) ]
        shell_mark = [f"{n:d}{angular_names[l]:s}" for l,n in enumerate(shell_count) ]

        f_out.write(f'#BASIS SET: ({",".join(prim_mark):s}) -> [{",".join(shell_mark):s}]\n')

        for bas_i in range(nbas):
            l = fakemol.bas_angular(bas_i)
            f_out.write(f"{atom:<2s}    {angular_names[l].upper():s}\n")
            exps = fakemol.bas_exp(bas_i)
            coeffs = fakemol.bas_ctr_coeff(bas_i)
            for exp, exp_coeff in zip(exps,coeffs):
                fmt = "{:>15.7G}" + "        {:>15.7G}"*len(exp_coeff) + "\n"
                f_out.write(fmt.format(exp, *exp_coeff).replace('E','D'))
    f_out.write('END\n')
