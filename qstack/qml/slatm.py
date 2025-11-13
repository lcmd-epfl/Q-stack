"""Spectrum of London and Axilrod-Teller-Muto potential (SLATM) representation.

Provides:
    - defaults: Default parameters for SLATM representation.
"""

import itertools
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from qstack.mathutils.array import stack_padding, vstack_padding


defaults = SimpleNamespace(sigma2=0.05, r0=0.1, rcut=4.8, dgrid2=0.03, theta0=20.0*np.pi/180.0, sigma3=0.05, dgrid3=0.03)


def get_mbtypes(qs, qml=False):
    """Generates many-body types (elements, pairs, triples) for SLATM representation.

    Args:
        qs (list): List of atomic number arrays for all molecules.
        qml (bool): If True, uses set ordering (QML-compatible). If False, uses sorted ordering. Defaults to False.

    Returns:
        dict: Dictionary with keys 1, 2, 3 containing:
        - 1: Array of unique elements
        - 2: List of element pairs (including self-pairs)
        - 3: List of valid element triples
    """
    # all the elements
    elements = itertools.chain.from_iterable(list(i) for i in qs)
    if qml:
        # bad because sets are not ordered
        elements = np.array(list(set(elements)))
    else:
        elements = np.unique(list(elements))

    # max number of atoms of each element across mols
    max_nq_in_mol = np.max([np.count_nonzero(q==elements[:,None], axis=1) for q in qs], axis=0)

    pairs = [(q, q) for q in elements] + [*itertools.combinations(elements, 2)]

    triples = []
    for q1 in elements:
        for (q2, q3) in pairs:
            for triple in [(q1, q2, q3), (q1, q3, q2), (q2, q1, q3)]:
                if (triple not in triples) and (triple[::-1] not in triples):
                    nq_in_triple = np.count_nonzero(triple==elements[:,None], axis=1)
                    if np.all(nq_in_triple <= max_nq_in_mol):
                        triples.append(triple)
    return {1: elements, 2: pairs, 3: triples}


def get_two_body(i, mbtype, q, dist,
                 r0=defaults.r0, rcut=defaults.rcut,
                 sigma=defaults.sigma2, dgrid=defaults.dgrid2):
    """Computes two-body London dispersion contribution for atom i.

    Evaluates the two-body term from pairwise 1/r^6 London dispersion interactions,
    projected onto a radial grid with Gaussian broadening of interatomic distances.

    Args:
        i (int): Index of the central atom.
        mbtype (tuple): Element pair (q1, q2) defining the two-body interaction type.
        q (numpy.ndarray): Array of atomic numbers for all atoms in molecule.
        dist (numpy.ndarray): Pairwise distance matrix (natom,natom) in Å.
        r0 (float): Minimum radial distance for grid. Defaults to 0.1 Å.
        rcut (float): Radial cutoff distance. Defaults to 4.8 Å.
        sigma (float): Gaussian width for distance broadening. Defaults to 0.05 Å.
        dgrid (float): Grid spacing for radial discretization. Defaults to 0.03 Å.

    Returns:
        numpy.ndarray: Two-body contribution on radial grid (ngrid,).
    """

    ngrid = int((rcut - r0)/dgrid) + 1
    rgrid = np.linspace(r0, rcut, ngrid)

    qi = q[i]
    if qi not in mbtype:
        return np.zeros_like(rgrid)

    if qi==mbtype[0]:
        (q1, q2) = mbtype
    else:
        (q1, q2) = mbtype[::-1]

    london = q1 * q2 / rgrid**6
    delta_norm = 1.0/(sigma * np.sqrt(2*np.pi))
    deltas = np.zeros(ngrid)

    j = np.where(q==q2)[0]
    j = j[j!=i]

    dist_ij = dist[np.ix_(j,[i])]
    dist_ij = dist_ij[np.where(dist_ij<rcut)[0]]

    delta = delta_norm * np.exp(-(rgrid-dist_ij)**2 * 0.5/(sigma**2))
    deltas += np.sum(delta, axis=0)

    return 0.5 * dgrid * london * deltas


def get_three_body(j, mbtype, q, r, dist,
                   rcut=defaults.rcut, theta0=defaults.theta0,
                   sigma=defaults.sigma3, dgrid=defaults.dgrid3):
    """Computes three-body Axilrod-Teller-Muto contribution for atom j.

    Evaluates the three-body ATM term from triple-dipole interactions,
    projected onto an angular grid with Gaussian broadening of bond angles.

    Args:
        j (int): Index of the central atom in the triplet.
        mbtype (tuple): Element triple (q1, q2, q3) defining the three-body interaction type.
        q (numpy.ndarray): Array of atomic numbers for all atoms.
        r (numpy.ndarray): Atomic position array (natom,3) in Å.
        dist (numpy.ndarray): Pairwise distance matrix (natom,natom) in Å.
        rcut (float): Distance cutoff for triplet formation. Defaults to 4.8 Å.
        theta0 (float): Margin for angular grid in radians. Defaults to 20°.
        sigma (float): Gaussian width for angle broadening in radians. Defaults to 0.05.
        dgrid (float): Grid spacing for angular discretization in radians. Defaults to 0.03.

    Returns:
        numpy.ndarray: Three-body contribution on angular grid (ngrid,).
    """

    def get_cos(a, b, c):
        """Computes cosine of angle abc from atomic positions.

        Args:
            a (int): Index of first atom.
            b (int): Index of vertex atom.
            c (int): Index of third atom.

        Returns:
            float: Cosine of angle abc.
        """
        v1 = r[a] - r[b]
        v2 = r[c] - r[b]
        return v1 @ v2 / (dist[a,b] * dist[b,c])

    theta1 = np.pi + theta0
    ngrid = int((theta1+theta0)/dgrid) + 1
    tgrid = np.linspace(-theta0, theta1, ngrid)
    spectrum = np.zeros_like(tgrid)

    (q1, q2, q3) = mbtype
    if q[j] != q2 or q1 not in q or q3 not in q:
        return spectrum

    delta_norm = 1.0/(sigma * np.sqrt(2*np.pi))

    for i in np.where(q==q1)[0]:
        if i==j or dist[i,j]>rcut:
            continue

        for k in np.where(q==q3)[0]:
            if k in [i,j] or dist[i,k]>rcut or dist[j,k]>rcut:
                continue

            cos_ikj = get_cos(i, k, j)
            cos_jik = get_cos(j, i, k)
            atm = (1.0 + np.cos(tgrid) * cos_ikj * cos_jik) / (dist[i,j]*dist[i,k]*dist[j,k])**3

            cos_ijk = get_cos(i, j, k)
            theta_ijk = np.arccos(cos_ijk)
            delta = delta_norm * np.exp(-(tgrid-theta_ijk)**2 * 0.5/(sigma**2))

            spectrum += delta * atm

    if q1==q3:
        spectrum *= 0.5
    return spectrum * dgrid * q1 * q2 * q3 / 3.0


def get_slatm(q, r, mbtypes, qml_compatible=True, stack_all=True,
              global_repr=False,
              r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2,
              theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3):
    """Computes SLATM representation for a single molecule.

    Constructs the SLATM (Spectrum of London and Axilrod-Teller-Muto potential)
    representation by combining one-body (nuclear charges), two-body (London dispersion),
    and three-body (Axilrod-Teller-Muto) contributions.

    Reference:
        B. Huang, O. A. von Lilienfeld,
        "Quantum machine learning using atom-in-molecule-based fragments selected on the fly",
        Nat. Chem. 12, 945–951 (2020), doi:10.1038/s41557-020-0527-z

    Args:
        q (numpy.ndarray): Array of atomic numbers (natom,).
        r (numpy.ndarray): Atomic position array (natom,3) in Å.
        mbtypes (dict): Many-body types from get_mbtypes with keys 1, 2, 3.
        qml_compatible (bool): If True, maintains QML package compatibility.
            If False, uses condensed representation (less 0s). Defaults to True.
            Is set to True if global_repr is True.
        stack_all (bool): If True, stacks all representations into one array.
            Defaults to True.
        global_repr (bool): If True, returns molecular SLATM (sum over atoms).
            If False, returns atomic SLATM. Defaults to False.
        r0 (float): Minimum radial distance for 2-body grid. Defaults to 0.1 Å.
        rcut (float): Radial cutoff for 2-body and 3-body terms. Defaults to 4.8 Å.
        sigma2 (float): Gaussian width for 2-body term. Defaults to 0.05 Å.
        dgrid2 (float): Grid spacing for 2-body term. Defaults to 0.03 Å.
        theta0 (float): Minimum angle for 3-body grid in radians. Defaults to 20°.
        sigma3 (float): Gaussian width for 3-body term in radians. Defaults to 0.05.
        dgrid3 (float): Grid spacing for 3-body term in radians. Defaults to 0.03.

    Returns:
        numpy.ndarray or dict: SLATM representation.
        - If stack_all=True and global_repr=False, numpy ndarray of shape (natom,n_features).
        - If global_repr=True, numpy ndarray of shape (n_features,).
        - If stack_all=False, returns dict with keys 1, 2, 3 containing lists of numpy ndarrays.
    """
    # for global representation, qml_compatible should be True
    qml_compatible = qml_compatible or global_repr

    natoms = len(q)
    dist = np.zeros((natoms, natoms))
    for (i, j) in itertools.combinations_with_replacement(range(natoms), 2):
        dist[i,j] = dist[j,i] = np.linalg.norm(r[i]-r[j])

    slatm = []
    for i, qi in enumerate(q):

        # 1-body terms
        if qml_compatible:
            slatm1b = (mbtypes[1] == qi)*qi.astype(float)
        else:
            slatm1b = np.array((float(qi),))

        # 2-body terms
        slatm2b = []
        for mbtype in mbtypes[2]:
            if (not qml_compatible) and (qi not in mbtype):
                continue
            two_body = get_two_body(i, mbtype, q, dist,
                                    sigma=sigma2, dgrid=dgrid2, r0=r0, rcut=rcut)
            slatm2b.append(two_body)

        # 3-body terms
        slatm3b = []
        for mbtype in mbtypes[3]:
            if (not qml_compatible) and (qi != mbtype[1]):
                continue
            three_body = get_three_body(i, mbtype, q, r, dist,
                                        sigma=sigma3, dgrid=dgrid3, rcut=rcut, theta0=theta0)
            slatm3b.append(three_body)

        # concatenate
        if stack_all:
            slatm2b = np.hstack(slatm2b)
            slatm3b = np.hstack(slatm3b)
            slatm.append(np.hstack((slatm1b, slatm2b, slatm3b)))
        else:
            slatm.append({1: slatm1b, 2: slatm2b, 3: slatm3b})

    if stack_all or global_repr:
        slatm = stack_padding(slatm)

    if global_repr:
        slatm = np.sum(slatm, axis=0)

    return slatm


def get_slatm_for_dataset(molecules,
                          progress=False,
                          global_repr=False,
                          qml_mbtypes=True, qml_compatible=True, stack_all=True,
                          r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2,
                          theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3):
    """Computes the (a)SLATM representation for a set of molecules.

    Generates SLATM descriptors for molecular datasets, automatically determining
    many-body types from all molecules.

    Args:
        molecules (Union[List[Mol], List[str]]): Pre-loaded molecules or paths
            to XYZ files. Mol can be any type with .numbers and .positions (Å) attributes,
            for example ASE Atoms objects.
        progress (bool): If True, displays progress bar. Defaults to False.
        global_repr (bool): If True, returns molecular SLATM (sum over atoms).
            If False, returns atomic SLATM (aSLATM). Defaults to False.
        qml_mbtypes (bool): If True, uses element ordering compatible with QML package
            (https://www.qmlcode.org/). If False, uses sorted ordering. Defaults to True.
        qml_compatible (bool): If False, uses condensed representation for local
            (global_repr=False) mode. Defaults to True.
        stack_all (bool): If True, stacks representations into one array. Defaults to True.
        r0 (float): Minimum radial distance for 2-body grid in Å. Defaults to 0.1.
        rcut (float): Radial cutoff for 2-body and 3-body terms in Å. Defaults to 4.8.
        sigma2 (float): Gaussian width for 2-body term in Å. Defaults to 0.05.
        dgrid2 (float): Grid spacing for 2-body term in Å. Defaults to 0.03.
        theta0 (float): Minimum angle for 3-body grid in radians. Defaults to 20° (π/9).
        sigma3 (float): Gaussian width for 3-body term in radians. Defaults to 0.05.
        dgrid3 (float): Grid spacing for 3-body term in radians. Defaults to 0.03.

    Returns:
        numpy.ndarray or List[List[numpy.ndarray]]: SLATM representations for all molecules.
            - If stack_all=True and global_repr=True, np.ndarray of shape (n_molecules, n_features),
            - If stack_all=True and global_repr=False, np.ndarray of shape (n_atoms_total, n_features),
            - If stack_all=False and global_repr=True, list of np.ndarrays of shape (n_features,) per molecule,
            - If stack_all=False and global_repr=False, list of lists of dicts per molecule with keys (1,2,3).
    """
    if isinstance(molecules[0], str):
        import ase.io
        molecules = [ase.io.read(xyz) for xyz in molecules]

    qs = [mol.numbers for mol in molecules]
    mbtypes = get_mbtypes(qs, qml=qml_mbtypes)

    slatm = [get_slatm(mol.numbers, mol.positions, mbtypes,
                       global_repr=global_repr,
                       qml_compatible=qml_compatible, stack_all=stack_all,
                       r0=r0, rcut=rcut, sigma2=sigma2, dgrid2=dgrid2,
                       theta0=theta0, sigma3=sigma3, dgrid3=dgrid3)
             for mol in tqdm(molecules, disable=not progress)]

    if stack_all:
        slatm = vstack_padding(slatm)

    return slatm


def get_slatm_rxn(reactions, progress=False, qml_mbtypes=True,
                  r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2,
                  theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3):
    """Computes the SLATM_d representation for chemical reactions.

    Calculates reaction representations as the difference between product and reactant
    SLATM descriptors (ΔR = R_products - R_reactants), suitable for predicting
    reaction properties like barriers and energies.

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
        progress (bool): If True, displays progress bar. Defaults to False.
        qml_mbtypes (bool): If True, uses element ordering compatible with QML package
            (https://www.qmlcode.org/). If False, uses sorted ordering. Defaults to True.
        r0 (float): Minimum radial distance for 2-body grid in Å. Defaults to 0.1.
        rcut (float): Radial cutoff for 2-body and 3-body terms in Å. Defaults to 4.8.
        sigma2 (float): Gaussian width for 2-body term in Å. Defaults to 0.05.
        dgrid2 (float): Grid spacing for 2-body term in Å. Defaults to 0.03.
        theta0 (float): Minimum angle for 3-body grid in radians. Defaults to 20° (π/9).
        sigma3 (float): Gaussian width for 3-body term in radians. Defaults to 0.05.
        dgrid3 (float): Grid spacing for 3-body term in radians. Defaults to 0.03.

    Returns:
        numpy.ndarray: SLATM_d difference representations of shape (n_reactions, n_features),
            where each row is the difference between product and reactant SLATM vectors.
    """
    qs = [mol.numbers for rxn in reactions for mol in rxn.reactants]
    mbtypes = get_mbtypes(qs, qml=qml_mbtypes)

    slatm_diff = []
    for reaction in tqdm(reactions, disable=not progress):
        slatm_reactants, slatm_products = [
                sum(get_slatm(mol.numbers, mol.positions, mbtypes,
                              global_repr=True, stack_all=True,
                              r0=r0, rcut=rcut, sigma2=sigma2, dgrid2=dgrid2,
                              theta0=theta0, sigma3=sigma3, dgrid3=dgrid3,
                              ) for mol in mols)
                for mols in (reaction.reactants, reaction.products)]
        slatm_diff.append(slatm_products-slatm_reactants)

    return np.vstack(slatm_diff)
