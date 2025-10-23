from collections import Counter
import numpy as np
from tqdm import trange


def get_global_K(X, Y, sigma, local_kernel, global_kernel, options, verbose=0):
    """

    .. todo::
        Write the docstring
    """
    n_x = len(X)
    n_y = len(Y)

    XY = np.concatenate((X, Y), axis=0)
    species = np.unique(XY[:,:,0].flatten())   # sorted by default

    mol_counts = [Counter(m[:,0]) for m in XY]
    max_atoms = {q: max(mol_counts, key=lambda x: x[q])[q] for q in species}
    max_size = sum(max_atoms.values())

    if verbose:
        print(f'{max_atoms=} {max_size=}')
        print("Computing global kernel elements:", flush=True)

    K_global = np.zeros((n_x, n_y))

    if Y is X:
        self = True
    else:
        self = False
        self_X = []
        self_Y = []

    for m in trange(0, n_x, disable=not verbose):
        if not self:
            K_self = get_covariance(X[m], X[m], max_atoms, local_kernel, sigma=sigma)
            self_X.append(global_kernel(K_self, options))
        for n in range(0, n_y):
            if not self:
                K_self = get_covariance(Y[m], Y[m], max_atoms, local_kernel, sigma=sigma)
                self_Y.append(global_kernel(K_self, options))
            K_pair = get_covariance(X[m], Y[n], max_atoms, local_kernel, sigma=sigma)
            K_global[m][n] = global_kernel(K_pair, options)
    if options['normalize'] == True:
        if self :
            K_global = normalize_kernel(K_global, self_x=None, self_y=None)
        else:
            K_global = normalize_kernel(K_global, self_x=self_X, self_y=self_Y)
    if verbose:
        print(f"Final global kernel has size : {K_global.shape}", flush=True)
    return K_global


def get_covariance(mol1, mol2, max_sizes, kernel , sigma=None):
    """

    .. todo::
        Write the docstring
    """
    species = sorted(max_sizes.keys())
    mol1_dict = mol_to_dict(mol1, species)
    mol2_dict = mol_to_dict(mol2, species)
    max_size = sum(max_sizes.values())
    K_covar = np.zeros((max_size, max_size))
    idx = 0
    for s in species:
        n1 = len(mol1_dict[s])
        n2 = len(mol2_dict[s])
        s_size = max_sizes[s]
        if n1 == 0 or n2 == 0:
            idx += s_size
            continue
        x1 = np.pad(mol1_dict[s], ((0, s_size - n1),(0,0)), 'constant')
        x2 = np.pad(mol2_dict[s], ((0, s_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+s_size, idx:idx+s_size] = kernel(x1, x2,  sigma)
        idx += s_size
    return K_covar


def normalize_kernel(kernel, self_x=None, self_y=None):
    """

    .. todo::
        Write the docstring
    """
    print("Normalizing kernel.")
    if self_x == None and self_y == None:
        self_cov = np.diag(kernel).copy()
        self_x = self_cov
        self_y = self_cov
    for n in range(kernel.shape[0]):
        for m in range(kernel.shape[1]):
            kernel[n][m] /= np.sqrt(self_x[n]*self_y[m])
    return kernel


def mol_to_dict(mol, species):
    """

    .. todo::
        Write the docstring
    """

    mol_dict = {s:[]  for s in species}
    for a in mol:
        mol_dict[a[0]].append(a[1])
    for k in mol_dict.keys():
        mol_dict[k] = np.array(mol_dict[k])
    return mol_dict


def avg_kernel(kernel, options):
    """

    .. todo::
        Write the docstring
    """
    avg = np.sum(kernel) / (kernel.shape[0] * kernel.shape[1])
    return avg


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
            error = np.sum((u - u_prev) ** 2) / np.sum(u ** 2) + np.sum((v - v_prev) ** 2) / np.sum(v **2)

        niter += 1
    p_alpha = np.multiply(np.multiply(K, u.reshape((-1, 1))) , v)
    K_rem = np.sum(np.multiply(p_alpha, kernel))
    return K_rem


global_kernels_dict = {
        'avg': avg_kernel,
        'rem': rematch_kernel,
        }
