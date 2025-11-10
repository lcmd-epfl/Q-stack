"""Bond-based reaction representation (B2R2) for chemical reactions.

Provides:
    defaults: default parameters for B2R2 computation.
"""

import itertools
from types import SimpleNamespace
import numpy as np
from scipy.special import erf
from tqdm import tqdm


defaults = SimpleNamespace(rcut=3.5, gridspace=0.03)


def get_bags(unique_ncharges):
    """Generate all unique element pair combinations including self-interactions.

    Args:
        unique_ncharges (array-like): Array of unique atomic charges/numbers.

    Returns:
        list: List of all unique element pairs [Z_i, Z_j] including self-interactions.
    """
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    self_combs = [[x, x] for x in unique_ncharges]
    combs += self_combs
    return combs


def get_mu_sigma(R):
    """Get Gaussian distribution parameters from interatomic distance.

    The constants used here are taken from the original B2R2 implementation.

    Args:
        R (float): Interatomic distance.

    Returns:
        tuple: Mean (mu) and standard deviation (sigma) for the Gaussian distribution.
    """
    mu = R * 0.5
    sigma = R * 0.125
    return mu, sigma


def get_gaussian(x, R):
    """Compute Gaussian function values for a given interatomic distance.

    Args:
        x (numpy ndarray): Grid points to evaluate the Gaussian.
        R (float): Interatomic distance determining the Gaussian parameters.

    Returns:
        numpy ndarray: Gaussian function values at the grid points.
    """
    mu, sigma = get_mu_sigma(R)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    return g


def get_skew_gaussian_l_both(x, R, Z_I, Z_J):
    """Compute skewed Gaussian distributions for B2R2_l representation.

    Args:
        x (numpy ndarray): Grid points to evaluate the functions.
        R (float): Interatomic distance.
        Z_I (int): Atomic number of atom I.
        Z_J (int): Atomic number of atom J.

    Returns:
        tuple: Two skewed Gaussian distributions (a, b) for the atom pair.
    """
    mu, sigma = get_mu_sigma(R)
    # a = Z_J * scipy.stats.skewnorm.pdf(x, Z_J, mu, sigma)
    # b = Z_I * scipy.stats.skewnorm.pdf(x, Z_I, mu, sigma)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    a = Z_J * g * e
    if Z_I==Z_J:
        return a, a
    e = 1.0 + erf(Z_I * X)
    b = Z_I * g * e
    return a, b


def get_skew_gaussian_n_both(x, R, Z_I, Z_J):
    """Compute combined skewed Gaussian distribution for B2R2_n representation.

    Args:
        x (numpy ndarray): Grid points to evaluate the function.
        R (float): Interatomic distance.
        Z_I (int): Atomic number of atom I.
        Z_J (int): Atomic number of atom J.

    Returns:
        numpy ndarray: Combined skewed Gaussian distribution for the atom pair.
    """
    mu, sigma = get_mu_sigma(R)
    # a = Z_I * scipy.stats.skewnorm.pdf(x, Z_J, mu, sigma)
    # b = Z_J * scipy.stats.skewnorm.pdf(x, Z_I, mu, sigma)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    e = 1.0 + erf(Z_J * X)
    a = Z_I * g * e
    if Z_I==Z_J:
        return 2.0*a
    e = 1.0 + erf(Z_I * X)
    b = Z_J * g * e
    return a + b


def get_b2r2_n_molecular(ncharges, coords, elements,
                         rcut=defaults.rcut, gridspace=defaults.gridspace):
    """Compute B2R2_n representation for a single molecule.

    Args:
        ncharges (array-like): Atomic numbers for all atoms in the molecule.
        coords (array-like): Atomic coordinates in Å, shape (natom, 3).
        elements (array-like): Unique atomic numbers present in the dataset.
        rcut (float): Cutoff radius for bond detection in Å. Defaults to 3.5.
        gridspace (float): Grid spacing for discretization in Å. Defaults to 0.03.

    Returns:
        numpy.ndarray: B2R2_n representation (ngrid,).
    """
    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    grid = np.arange(0, rcut, gridspace)
    twobodyrep = np.zeros_like(grid)

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < rcut:
                twobodyrep += get_skew_gaussian_n_both(grid, R, ncharge_b, ncharge_a)

    return twobodyrep


def get_b2r2_a_molecular(ncharges, coords, elements,
                         rcut=defaults.rcut, gridspace=defaults.gridspace):
    """Compute B2R2_a representation for a single molecule.

    Args:
        ncharges (array-like): Atomic numbers for all atoms in the molecule.
        coords (array-like): Atomic coordinates in Å, shape (natom, 3).
        elements (array-like): Unique atomic numbers present in the dataset.
        rcut (float): Cutoff radius for bond detection in Å. Defaults to 3.5.
        gridspace (float): Grid spacing for discretization in Å. Defaults to 0.03.

    Returns:
        numpy.ndarray: B2R2_a representation (n_pairs*ngrid,).
    """
    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    bags = get_bags(elements)
    grid = np.arange(0, rcut, gridspace)
    twobodyrep = np.zeros((len(bags), len(grid)))

    bag_idx      = {tuple(q1q2): i for i, q1q2 in enumerate(bags)}
    bag_idx.update({tuple(q1q2[::-1]): i for i, q1q2 in enumerate(bags)})

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < rcut:
                twobodyrep[bag_idx[ncharge_a, ncharge_b]] += get_gaussian(grid, R)

    twobodyrep = 2.0*np.concatenate(twobodyrep)
    return twobodyrep


def get_b2r2_l_molecular(ncharges, coords, elements,
                         rcut=defaults.rcut, gridspace=defaults.gridspace):
    """Compute B2R2_l representation for a single molecule.

    Args:
        ncharges (array-like): Atomic numbers for all atoms in the molecule.
        coords (array-like): Atomic coordinates in Å, shape (natom, 3).
        elements (array-like): Unique atomic numbers present in the dataset.
        rcut (float): Cutoff radius for bond detection in Å. Defaults to 3.5.
        gridspace (float): Grid spacing for discretization in Å. Defaults to 0.03.

    Returns:
        numpy.ndarray: B2R2_l representation (n_elements*ngrid,).
    """
    idx_relevant_atoms = np.where(np.sum(np.array(ncharges)==np.array(elements)[:,None], axis=0))
    ncharges = np.array(ncharges)[idx_relevant_atoms]
    coords = np.array(coords)[idx_relevant_atoms]

    bags = np.array(elements)
    grid = np.arange(0, rcut, gridspace)
    twobodyrep = np.zeros((len(bags), len(grid)))

    bag_idx = {q: i for i,q in enumerate(bags)}

    for i, ncharge_a in enumerate(ncharges):
        for j, ncharge_b in enumerate(ncharges[:i]):
            coords_a = coords[i]
            coords_b = coords[j]
            R = np.linalg.norm(coords_b - coords_a)
            if R < rcut:
                a, b = get_skew_gaussian_l_both(grid, R, ncharge_a, ncharge_b)
                twobodyrep[bag_idx[ncharge_a]] += a
                twobodyrep[bag_idx[ncharge_b]] += b

    twobodyrep = np.concatenate(twobodyrep)
    return twobodyrep


def get_b2r2(reactions, variant='l', progress=False, rcut=defaults.rcut, gridspace=defaults.gridspace):
    """High-level interface for computing bond-based reaction representations (B2R2).

    Reference:
        P. van Gerwen, A. Fabrizio, M. D. Wodrich, C. Corminboeuf,
        "Physics-based representations for machine learning properties of chemical reactions",
        Mach. Learn.: Sci. Technol. 3, 045005 (2022), doi:10.1088/2632-2153/ac8f1a

    Args:
        reactions (List[rxn]): List of reaction objects with attributes:
            - rxn.reactants (List[Mol]): List of reactant molecules.
            - rxn.products (List[Mol]): List of product molecules.
            Mol can be any type with .numbers and .positions (Å) attributes,
            for example ASE Atoms objects.
        variant (str): B2R2 variant to compute. Options:
            - 'l': Local variant with element-resolved skewed Gaussians (default).
            - 'a': Agnostic variant with element-pair Gaussians.
            - 'n': Nuclear variant with combined skewed Gaussians.
        progress (bool): If True, displays progress bar. Defaults to False.
        rcut (float): Cutoff radius for bond detection in Å. Defaults to 3.5.
        gridspace (float): Grid spacing for discretization in Å. Defaults to 0.03.

    Returns:
        numpy.ndarray: B2R2 representations of shape (n_reactions, n_features).
            For variants 'l' and 'a', returns difference (products - reactants).
            For variant 'n', returns concatenation [reactants, products].

    Raises:
        RuntimeError: If an unknown variant is specified.
    """
    if variant=='l':
        get_b2r2_molecular=get_b2r2_l_molecular
        combine = lambda r,p: p-r
    elif variant=='a':
        get_b2r2_molecular = get_b2r2_a_molecular
        combine = lambda r,p: p-r
    elif variant=='n':
        get_b2r2_molecular=get_b2r2_n_molecular
        combine = lambda r,p: np.hstack((r,p))
    else:
        raise RuntimeError(f'Unknown B2R2 {variant=}')
    return get_b2r2_inner(reactions, progress=progress, rcut=rcut, gridspace=gridspace,
                          get_b2r2_molecular=get_b2r2_molecular, combine=combine)


def get_b2r2_inner(reactions,
                   progress=False,
                   rcut=defaults.rcut, gridspace=defaults.gridspace,
                   get_b2r2_molecular=None, combine=None):
    """Compute the B2R2 representations for a list of reactions.

    Internal implementation function that computes B2R2 representations using
    provided molecular representation function and combination strategy.
    Automatically determines element set from all reactant molecules.

    Args:
        reactions (List[rxn]): List of reaction objects with attributes:
            - rxn.reactants (List[Mol]): List of reactant molecules.
            - rxn.products (List[Mol]): List of product molecules.
            Mol can be any type with .numbers and .positions (Å) attributes,
            for example ASE Atoms objects.
        progress (bool): If True, displays progress bar. Defaults to False.
        rcut (float): Cutoff radius for bond detection in Å. Defaults to 3.5.
        gridspace (float): Grid spacing for discretization in Å. Defaults to 0.03.
        get_b2r2_molecular (callable): Function to compute molecular representations.
            Should be one of get_b2r2_{l,n,a}_molecular.
        combine (callable): Function(r: ndarray, p: ndarray) -> ndarray to combine
            reactant and product representations (e.g., difference or concatenation).

    Returns:
        numpy.ndarray: B2R2 representations of shape (n_reactions, n_features),
            where each row represents a reaction according to the combine function.
    """
    qs = [mol.numbers for rxn in reactions for mol in rxn.reactants]
    elements = np.unique(np.concatenate(qs))

    b2r2_diff = []
    for reaction in tqdm(reactions, disable=not progress):
        b2r2_reactants, b2r2_products = [
                sum(get_b2r2_molecular(mol.numbers, mol.positions,
                                       rcut=rcut,
                                       gridspace=gridspace,
                                       elements=elements,
                                       ) for mol in mols)
                for mols in (reaction.reactants, reaction.products)]
        b2r2_diff.append(combine(b2r2_reactants, b2r2_products))

    return np.vstack(b2r2_diff)
