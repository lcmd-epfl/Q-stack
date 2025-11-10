"""Global (molecular) kernel implementations.

Provides:
    global_kernels_dict: Dictionary mapping global kernel names to functions.
"""

import math
from collections import Counter
import numpy as np
from tqdm import tqdm


def get_global_K(X, Y, sigma, local_kernel, global_kernel, options):
    """Compute global kernel matrix between two sets of molecular representations.

    Args:
        X (list): List of molecular representations (first set).
        Y (list): List of molecular representations (second set).
        sigma (float): Kernel width parameter.
        local_kernel (callable): Local kernel function for atomic environments.
        global_kernel (callable): Global kernel function for combining local kernels.
        options (dict): Dictionary of global kernel options.

    Returns:
        numpy ndarray: Global kernel matrix of shape (len(X), len(Y)).
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
    """Compute the covariance matrix between two molecules using local kernels.

    Args:
        mol1 (dict): First molecule represented as dictionary of atomic environments by species.
        mol2 (dict): Second molecule represented as dictionary of atomic environments by species.
        species (numpy ndarray): Array of unique atomic species present in the dataset.
        max_atoms (dict): Maximum number of atoms per species across all molecules.
        max_size (int): Total size of the padded covariance matrix.
        kernel (callable): Local kernel function.
        sigma (float, optional): Kernel width parameter. Defaults to None.

    Returns:
        numpy ndarray: Covariance matrix of shape (max_size, max_size).
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
    """Normalize a kernel matrix using self-kernel values.

    Args:
        kernel (numpy ndarray): Kernel matrix to normalize.
        self_x (numpy ndarray, optional): Self-kernel values for X. If None, extracted from diagonal. Defaults to None.
        self_y (numpy ndarray, optional): Self-kernel values for Y. If None, extracted from diagonal. Defaults to None.
        verbose (int): Verbosity level. Defaults to 0.

    Returns:
        numpy ndarray: Normalized kernel matrix.
    """
    if verbose:
        print("Normalizing kernel.")
    if self_x is None and self_y is None:
        self_cov = np.diag(kernel).copy()
        self_x = self_cov
        self_y = self_cov
    return kernel / np.sqrt(np.outer(self_x, self_y))


def mol_to_dict(mol, species):
    """Convert molecular representation to a dictionary organized by atomic species.

    Args:
        mol (numpy ndarray): Molecular representation where each row is [atomic_number, features...].
        species (numpy ndarray): Array of unique atomic species.

    Returns:
        dict: Dictionary mapping atomic numbers to arrays of atomic feature vectors.
    """
    mol_dict = {q:[] for q in species}
    for atom in mol:
        mol_dict[atom[0]].append(atom[1])
    for q in mol_dict:
        mol_dict[q] = np.array(mol_dict[q])
    return mol_dict


def sumsq(x):
    """Compute sum of squares (dot product with itself).

    Args:
        x (numpy ndarray): Input vector.

    Returns:
        float: Sum of squared elements.
    """
    return x@x


def avg_kernel(kernel, _options):
    """Compute the average kernel value.

    Args:
        kernel (numpy ndarray): Kernel matrix.
        _options (dict): Options dictionary (unused).

    Returns:
        float: Average of all kernel matrix elements.
    """
    return np.sum(kernel) / math.prod(kernel.shape)


def rematch_kernel(kernel, options):
    """Compute the REMatch (Regularized Entropy Match) kernel.

    Uses Sinkhorn algorithm to compute optimal transport-based kernel similarity.

    Reference:
        S. De, A. P. Bartók, G. Csányi, M. Ceriotti,
        "Comparing molecules and solids across structural and alchemical space",
        Phys. Chem. Chem. Phys. 18, 13754 (2016), doi:10.1039/C6CP00415F

    Args:
        kernel (numpy ndarray): Local kernel matrix.
        options (dict): Options dictionary containing 'alpha' parameter for regularization.

    Returns:
        float: REMatch kernel value.
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
