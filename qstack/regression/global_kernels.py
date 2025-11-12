import math
from collections import Counter
import numpy as np
from tqdm import tqdm


def get_global_K(X, Y, sigma, local_kernel, global_kernel, options):
    """

    .. todo::
        Write the docstring
    """
    self = (Y is X)
    verbose = options.get('verbose', 0)

    XY = np.concatenate((X, Y), axis=0)
    species = np.unique(XY[:,:,0].flatten())   # sorted by default
    mol_counts = [Counter(m[:,0]) for m in XY]
    max_atoms = {q: max(mol_counts, key=lambda x: x[q])[q] for q in species}
    max_size = sum(max_atoms.values())

    if verbose:
        print(f'{max_atoms=} {max_size=}')
        print("Computing global kernel elements:", flush=True)

    X_dict = [mol_to_dict(x, species) for x in X]
    Y_dict = [mol_to_dict(y, species) for y in Y]
    K_global = np.zeros((len(X), len(Y)))

    for m, x in enumerate(tqdm(X_dict, disable=not verbose)):
        y_start = m if self else 0
        for n, y in enumerate(Y_dict[y_start:], start=y_start):
            K_pair = get_covariance(x, y, species, max_atoms, max_size, local_kernel, sigma=sigma)
            K_global[m,n] = global_kernel(K_pair, options)
            if self:
                K_global[n,m] = K_global[m,n]

    if options['normalize']:
        if self:
            K_global = normalize_kernel(K_global, verbose=verbose)
        else:
            self_X, self_Y = [], []
            for x in tqdm(X_dict, disable=not verbose):
                K_self = get_covariance(x, x, species, max_atoms, max_size, local_kernel, sigma=sigma)
                self_X.append(global_kernel(K_self, options))
            for y in tqdm(Y_dict, disable=not verbose):
                K_self = get_covariance(y, y, species, max_atoms, max_size, local_kernel, sigma=sigma)
                self_Y.append(global_kernel(K_self, options))
            K_global = normalize_kernel(K_global, self_x=self_X, self_y=self_Y, verbose=verbose)

    if verbose:
        print(f"Final global kernel has size : {K_global.shape}", flush=True)
    return K_global


def get_covariance(mol1, mol2, species, max_atoms, max_size, kernel, sigma=None):
    """

    .. todo::
        Write the docstring
    """
    K_covar = np.zeros((max_size, max_size))
    idx = 0
    for q in species:
        n1 = len(mol1[q])
        n2 = len(mol2[q])
        q_size = max_atoms[q]
        if n1==0 or n2==0:
            idx += q_size
            continue
        x1 = np.pad(mol1[q], ((0, q_size - n1),(0,0)), 'constant')
        x2 = np.pad(mol2[q], ((0, q_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+q_size, idx:idx+q_size] = kernel(x1, x2,  sigma)
        idx += q_size
    return K_covar


def normalize_kernel(kernel, self_x=None, self_y=None, verbose=0):
    """

    .. todo::
        Write the docstring
    """
    if verbose:
        print("Normalizing kernel.")
    if self_x is None and self_y is None:
        self_cov = np.diag(kernel).copy()
        self_x = self_cov
        self_y = self_cov
    return kernel / np.sqrt(np.outer(self_x, self_y))


def mol_to_dict(mol, species):
    """

    .. todo::
        Write the docstring
    """

    mol_dict = {q:[] for q in species}
    for atom in mol:
        mol_dict[atom[0]].append(atom[1])
    for q in mol_dict:
        mol_dict[q] = np.array(mol_dict[q])
    return mol_dict


def sumsq(x):
    return x@x


def avg_kernel(kernel, _options):
    """

    .. todo::
        Write the docstring
    """
    return np.sum(kernel) / math.prod(kernel.shape)


def rematch_kernel(kernel, options):
    """

    .. todo::
        Write the docstring
    """
    alpha = options['alpha']
    thresh = 1e-6
    n, m = kernel.shape

    K = np.exp(-(1 - kernel) / alpha)

    en = np.ones((n,)) / float(n)
    em = np.ones((m,)) / float(m)

    u = en.copy()
    v = em.copy()

    niter = 0
    error = 1

    while error > thresh:
        v_prev = v
        u_prev = u

        u = np.divide(en, np.dot(K, v))
        v = np.divide(em, np.dot(K.T, u))

        if niter % 5:
            error = sumsq(u-u_prev) / sumsq(u) + sumsq(v-v_prev) / sumsq(v)

        niter += 1
    p_alpha = np.multiply(np.multiply(K, u.reshape((-1, 1))) , v)
    K_rem = np.sum(np.multiply(p_alpha, kernel))
    return K_rem


global_kernels_dict = {
        'avg': avg_kernel,
        'rem': rematch_kernel,
        }
