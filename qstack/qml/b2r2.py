import itertools
from types import SimpleNamespace
import numpy as np
from scipy.special import erf


defaults = SimpleNamespace(rcut=3.5, gridspace=0.03)


def get_bags(unique_ncharges):
    combs = list(itertools.combinations(unique_ncharges, r=2))
    combs = [list(x) for x in combs]
    # add self interaction
    self_combs = [[x, x] for x in unique_ncharges]
    combs += self_combs
    return combs


def get_mu_sigma(R):
    mu = R * 0.5
    sigma = R * 0.125
    return mu, sigma


def get_gaussian(x, R):
    mu, sigma = get_mu_sigma(R)
    X = (x-mu) / (sigma*np.sqrt(2))
    g = np.exp(-X**2) / (np.sqrt(2*np.pi) * sigma)
    return g


def get_skew_gaussian_l_both(x, R, Z_I, Z_J):
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
                twobodyrep[bag_idx[(ncharge_a, ncharge_b)]] += get_gaussian(grid, R)

    twobodyrep = 2.0*np.concatenate(twobodyrep)
    return twobodyrep


def get_b2r2_l_molecular(ncharges, coords, elements,
                         rcut=defaults.rcut, gridspace=defaults.gridspace):

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


def get_b2r2(reactions, variant='l', rcut=defaults.rcut, gridspace=defaults.gridspace):
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
    return get_b2r2_inner(reactions, rcut=rcut, gridspace=gridspace,
                          get_b2r2_molecular=get_b2r2_molecular, combine=combine)


def get_b2r2_inner(reactions,
                   progress=False,
                   rcut=defaults.rcut, gridspace=defaults.gridspace,
                   get_b2r2_molecular=None, combine=None):

    """ Computes the B2R2 representations for a list of reactions.

    Reference:
        P. van Gerwen, A. Fabrizio, M. D. Wodrich, C. Corminboeuf,
        "Physics-based representations for machine learning properties of chemical reactions",
        Mach. Learn.: Sci. Technol. 3, 045005 (2022), doi:10.1088/2632-2153/ac8f1a.

    Args:
        reactions (List[rxn]): a list of rxn objects containing reaction information.
            rxn.reactants (List[ase.Atoms]) is a list of reactants (ASE molecules),
            rxn.products (List[ase.Atoms]) is a list of products.
        rcut (float): cutoff radius (Å)
        gridspace (float): grid spacing (Å)
        get_b2r2_molecular (func): function to compute the molecular representations,
                                   i.e. one of `get_b2r2_{l,n,a}_molecular`
        combine (func(r: ndarray, p: ndarray)): function to combine the reactants and products representations,
                                   e.g. difference or concatenation
    Returns:
        ndrarray containing the B2R2 representation for each reaction
    """

    qs = [mol.numbers for rxn in reactions for mol in rxn.reactants]
    elements = np.unique(np.concatenate(qs))

    if progress:
        import tqdm
        reactions = tqdm.tqdm(reactions)

    b2r2_diff = []
    for reaction in reactions:
        b2r2_reactants, b2r2_products = [
                sum(get_b2r2_molecular(mol.numbers, mol.positions,
                                       rcut=rcut,
                                       gridspace=gridspace,
                                       elements=elements,
                                       ) for mol in mols)
                for mols in (reaction.reactants, reaction.products)]
        b2r2_diff.append(combine(b2r2_reactants, b2r2_products))

    return np.vstack(b2r2_diff)
