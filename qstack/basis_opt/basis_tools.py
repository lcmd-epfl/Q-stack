import copy
import numpy as np
from pyscf import df, dft


def energy_mol(newbasis, moldata):
    """Computes energy for basis optimization.

    Args:
        newbasis (dict): New basis set definition.
        moldata (dict): Dictionary containing molecular data including mol, rho, coords, weights, and self.

    Returns:
        float: Energy value for the given basis.
    """
    mol     = moldata['mol'    ]
    rho     = moldata['rho'    ]
    coords  = moldata['coords' ]
    weights = moldata['weights']
    self    = moldata['self'   ]

    newmol = df.make_auxmol(mol, newbasis)
    ao = dft.numint.eval_ao(newmol, coords).T
    w = np.einsum('pi,i,i->p', ao, rho, weights)
    S = np.einsum('pi,qi,i->pq', ao, ao, weights)
    c = np.linalg.solve(S, w)
    E = self-c@w

    return E


def gradient_mol(nexp, newbasis, moldata):
    """Computes energy and gradient for basis optimization.

    Args:
        nexp (int): Number of exponents.
        newbasis (dict): New basis set definition.
        moldata (dict): Dictionary containing molecular data including mol, rho, coords, weights, 
            self, idx, centers, and distances.

    Returns:
        tuple: A tuple containing:
            - E (float): Energy value.
            - dE_da (numpy.ndarray): Gradient of energy with respect to exponents.
    """

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

    w = np.einsum('pi,i,i->p', ao, rho, weights)
    dw_da = np.zeros((nexp, newmol.nao))
    for p in range(newmol.nao):
        iat = centers[p]
        r2 = distances[iat]
        dw_da[idx[p], p] = np.einsum('i,i,i,i->', ao[p], rho, r2, weights)

    S = np.einsum('pi,qi,i->pq', ao, ao, weights)
    dS_da = np.zeros((nexp, newmol.nao, newmol.nao))

    for p in range(newmol.nao):
        for q in range(p, newmol.nao):
            ip = idx[p]
            iq = idx[q]
            iatp = centers[p]
            iatq = centers[q]
            r2p = distances[iatp]
            r2q = distances[iatq]
            ao_ao_w = np.einsum('i,i,i->i', ao[p], ao[q], weights)
            ip_p_q = np.einsum('i,i->', ao_ao_w, r2p)
            iq_p_q = np.einsum('i,i->', ao_ao_w, r2q)
            dS_da[ip, p, q] += ip_p_q
            dS_da[iq, p, q] += iq_p_q
            if p != q:
                dS_da[ip, q, p] += ip_p_q
                dS_da[iq, q, p] += iq_p_q

    c = np.linalg.solve(S, w)
    part1 = np.einsum('p,ip->i',  c, dw_da)
    part2 = np.einsum('p,ipq,q->i',  c, dS_da, c)
    dE_da = 2.0*part1 - part2
    E = self - c@w
    return E, dE_da


def exp2basis(exponents, elements, basis):
    """Convert exponents array to basis set format.

    Args:
        exponents (numpy.ndarray): Array of basis function exponents.
        elements (list): List of element symbols.
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
        bf_bounds (dict): Dictionary mapping elements to their index bounds.

    Returns:
        numpy.ndarray: Concatenated array containing only specified elements.
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
    for q, b in basis.items():
        print('  "'+q+'": [', file=f)
        for i, gto in enumerate(b):
            if i > 0:
                print(',', file=f)
            print('   ', gto, file=f, end='')
        print('  ]', file=f)
    print('}', file=f)
