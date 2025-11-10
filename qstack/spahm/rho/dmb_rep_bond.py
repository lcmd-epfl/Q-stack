"""Functions for SPAHM(b) computation."""

import operator
from ast import literal_eval
import numpy as np
from pyscf import gto
from qstack import fields
from . import sym, lowdin
from .Dmatrix import Dmatrix_for_z, c_split, rotate_c


def make_bname(q0, q1):
    """Create canonical bond name from two element symbols.

    Args:
        q0 (str): First element symbol.
        q1 (str): Second element symbol.

    Returns:
        str: Concatenated element symbols in alphabetical order (e.g., 'CH', 'CC', 'NO').
    """
    return operator.concat(*sorted((q0, q1)))


def get_basis_info(qqs, mybasis, only_m0, printlevel):
    """Compute basis indices and metric matrices for bond pairs.

    Args:
        qqs (list): List of bond pair names (e.g., ['CC', 'CH', 'OH']).
        mybasis (dict): Dictionary mapping bond names to basis set dictionaries.
        only_m0 (bool): If True, use only m=0 angular momentum components.
        printlevel (int): Verbosity level.

    Returns:
        tuple: (idx, M) where:
        - idx (dict): Pair indices for each bond type (list of [i, j] pairs)
        - M (dict): Metric matrices for each bond type (numpy 2D ndarray)
    """
    idx = {}
    M   = {}
    for qq in qqs:
        if printlevel>1:
            print(qq)
        S, ao, _ = sym.get_S('No', mybasis[qq])
        if not only_m0:
            idx[qq] = sym.store_pair_indices_z(ao)
        else:
            idx[qq] = sym.store_pair_indices_z_only0(ao)
        M[qq] = sym.metric_matrix_z(idx[qq], ao, S)
    return idx, M


def read_df_basis(bnames, bpath, same_basis=False):
    """Load bond-optimized basis sets from .bas files.

    Args:
        bnames (list): List of bond pair names (e.g., ['CC', 'CH']).
        bpath (str): Directory path containing .bas files.
        same_basis (bool): If True, uses generic CC.bas for all pairs. Defaults to False.

    Returns:
        dict: Dictionary mapping bond names to basis set dictionaries.
    """
    mybasis = {}
    for bname in bnames:
        if bname in mybasis:
            continue
        fname = f'{bpath}/{bname}.bas' if not same_basis else f'{bpath}/CC.bas'
        with open(fname) as f:
            mybasis[bname] = literal_eval(f.read())
    return mybasis


def get_element_pairs(elements):
    """Generate all possible element pair combinations.

    Creates complete list of bond types assuming all elements can bond with each other.

    Args:
        elements (list): List of element symbols.

    Returns:
        tuple: (qqs, qqs4q) where:
        - qqs (list): Sorted list of unique bond pair names
        - qqs4q (dict): Maps each element to its list of possible bond partners
    """
    qqs   = []
    qqs4q = {}
    for q1 in elements:
        qqs4q[q1] = []
        for q2 in elements:
            qq = make_bname(q1, q2)
            qqs4q[q1].append(qq)
            qqs.append(qq)
        qqs4q[q1].sort()
    qqs = sorted(set(qqs))
    return qqs, qqs4q


def get_element_pairs_cutoff(elements, mols, cutoff, align=False):
    """Determine element pairs based on actual distances in molecules.

    Identifies which element pairs actually form bonds within the distance cutoff
    by scanning molecular geometries.

    Args:
        elements (list): List of element symbols to consider.
        mols (list): List of pyscf Mole objects.
        cutoff (float): Maximum bond distance in Å.
        align (bool): If True, includes all element pairs regardless of distance.
            Defaults to False.

    Returns:
        tuple: (qqs, qqs4q) where:
        - qqs (list): Sorted list of bond pair names found within cutoff
        - qqs4q (dict): Maps each element to its list of bond partners
    """
    qqs4q = {q: [] for q in elements}
    qqs = []
    if align:
        for q0 in elements:
            for q1 in elements:
                qq = make_bname(q1,q0)
                qqs4q[q0].append(qq)
                qqs4q[q1].append(qq)
                qqs.append(qq)
    else:
        for mol in mols:
            q = [mol.atom_symbol(i) for i in range(mol.natm)]
            r = mol.atom_coords(unit='ANG')
            for i0, q0 in enumerate(q):
                if q0 not in elements:
                    continue
                for i1, q1 in enumerate(q[:i0]):
                    if q1 not in elements:
                        continue
                    if np.linalg.norm(r[i1]-r[i0]) <= cutoff:
                        qq = make_bname(q1,q0)
                        qqs4q[q0].append(qq)
                        qqs4q[q1].append(qq)
                        qqs.append(qq)
    qqs4q = {q: sorted(set(qqs4q[q])) for q in elements}
    qqs   = sorted(set(qqs))
    return qqs, qqs4q


def read_basis_wrapper_pairs(mols, bondidx, bpath, only_m0, printlevel, same_basis=False):
    """Read basis sets and computes metric matrices for specified bond pairs.

    Processes bond pairs from molecular structures and loads their corresponding
    basis sets from disk, then computes basis indices and metric matrices.

    Args:
        mols (list): List of pyscf Mole objects.
        bondidx (list): List of bond index pairs [(i0, i1), ...] for each molecule.
        bpath (str): Directory path containing basis set files.
        only_m0 (bool): If True, use only m=0 angular momentum components.
        printlevel (int): Verbosity level for output (>1 for detailed printing).
        same_basis (bool): If True, uses generic CC.bas for all pairs. Defaults to False.

    Returns:
        tuple: (mybasis, idx, M) where:
            - mybasis (dict): Bond name to basis set dictionary mapping,
            - idx (dict): Pair indices (list of [i, j] pairs) for each bond type,
            - M (dict): Metric matrices (2D numpy ndarray) for each bond type.
    """
    qqs0 = [make_bname(*map(mol.atom_symbol, bondij)) for (bondij, mol) in zip(bondidx, mols, strict=True)]
    qqs0 = sorted(set(qqs0))
    if printlevel>1:
        print(qqs0)
    mybasis = read_df_basis(qqs0, bpath, same_basis=same_basis)
    idx, M  = get_basis_info(qqs0, mybasis, only_m0, printlevel)
    return mybasis, idx, M


def read_basis_wrapper(mols, bpath, only_m0, printlevel, cutoff=None, elements=None, pairfile=None, dump_and_exit=False, same_basis=False):
    """Read basis sets for all element pairs present in molecules.

    Determines which element pairs exist (either all possible or within cutoff distance),
    loads corresponding basis sets, and computes metric matrices and indices.
    Can cache pair information to file for subsequent runs.

    Args:
        mols (list): List of pyscf Mole objects to analyze.
        bpath (str): Directory path containing basis set files.
        only_m0 (bool): If True, use only m=0 angular momentum components.
        printlevel (int): Verbosity level for output (>1 for detailed printing).
        cutoff (float, optional): Maximum bond distance in Å for pair detection.
            If None, considers all element combinations. Defaults to None.
        elements (list, optional): List of element symbols to consider. If None,
            extracts all elements from molecules. Defaults to None.
        pairfile (str, optional): Path to save/load element pair information. Defaults to None.
        dump_and_exit (bool): If True, saves pair information and exits. Defaults to False.
        same_basis (bool): If True, uses generic CC.bas for all pairs. Defaults to False.

    Returns:
        tuple: (elements, mybasis, qqs, qqs4q, idx, M) where:
            - elements (list): Sorted list of element symbols.
            - mybasis (dict): Bond name to basis set dictionary mapping.
            - qqs (dict): Maps each element to list of all bond pair names.
            - qqs4q (dict): Maps each element to its specific bond partners.
            - idx (dict): Pair indices (list of [i, j] pairs) for each bond type.
            - M (dict): Metric matrices (2D numpy ndarray) for each bond type.
    """
    if elements is None:
        elements = sorted({q for mol in mols for q in mol.elements})

    if pairfile and not dump_and_exit:
        qqs0, qqs4q = np.load(pairfile, allow_pickle=True)
    else:
        if cutoff is None:
            qqs0, qqs4q = get_element_pairs(elements)
        else:
            qqs0, qqs4q = get_element_pairs_cutoff(elements, mols, cutoff, align=True)

    if pairfile and dump_and_exit:
        np.save(pairfile, np.asanyarray((qqs0, qqs4q), dtype=object))
        exit(0)

    qqs = dict.fromkeys(elements, qqs0)
    if printlevel>1:
        print(qqs0)
    mybasis = read_df_basis(qqs0, bpath, same_basis=same_basis)
    idx, M  = get_basis_info(qqs0, mybasis, only_m0, printlevel)
    return elements, mybasis, qqs, qqs4q, idx, M


def bonds_dict_init(qqs, M):
    """Initialize storage for bond representations.

    Creates a dictionary with zero-initialized arrays for each bond type,
    with array sizes matching the corresponding metric matrix dimensions.

    Args:
        qqs (list): List of bond pair names (e.g., ['CC', 'CH', 'OH']).
        M (dict): Dictionary mapping bond names to metric matrices.

    Returns:
        tuple: (mybonds, N) where:
            - mybonds (dict): Bond name to zero-initialized numpy array mapping.
            - N (int): Total number of basis functions across all bond types.
    """
    N = 0
    mybonds = {}
    for qq in qqs:
        n = len(M[qq])
        mybonds[qq] = np.zeros(n)
        N += n
    return mybonds, N


def fit_dm(dm, mol, mybasis, ri0, ri1):
    """Fit density matrix using auxiliary basis functions at bond center.

    Decomposes the bond density matrix into auxiliary basis coefficients
    centered at the bond midpoint, then splits coefficients by angular momentum.

    Args:
        dm (numpy.ndarray): Density matrix for the bond.
        mol (pyscf.gto.Mole): Molecule object containing the bond.
        mybasis (dict): Basis set dictionary for the bond type.
        ri0 (numpy.ndarray): Coordinates of first atom in Å.
        ri1 (numpy.ndarray): Coordinates of second atom in Å.

    Returns:
        list: Coefficients split by angular momentum quantum number [(l, coeff), ...].
    """
    rm = (ri0+ri1)*0.5
    atom = f"No {rm[0]} {rm[1]} {rm[2]}"
    auxmol = gto.M(atom=atom, basis=mybasis)
    e2c, e3c = fields.decomposition.get_integrals(mol, auxmol)[1:]
    c = fields.decomposition.get_coeff(dm, e2c, e3c)
    cs = c_split(auxmol, c)
    return cs


def vec_from_cs(z, cs, lmax, idx):
    """Rotate basis coefficients to bond axis and creates vectorized representation.

    Applies Wigner D-matrix rotation to align coefficients with the bond vector
    (same as pretending the bond is along the z-axis),
    ensuring rotational invariance, then vectorizes using symmetry indices.

    Args:
        z (numpy.ndarray): Bond vector (displacement from one atom to another).
        cs (list): Angular momentum decomposed coefficients [(l, coeff), ...].
        lmax (int): Maximum angular momentum quantum number.
        idx (dict): Pair indices mapping for symmetrization.

    Returns:
        numpy.ndarray: Rotationally invariant vectorized representation of the bond.
    """
    D = Dmatrix_for_z(z, lmax)
    c_new = rotate_c(D, cs)
    v = sym.vectorize_c(idx, c_new)
    return v


def repr_for_bond(i0, i1, L, mybasis, idx, q, r, cutoff):
    """Compute bond representation for a specific atom pair.

    Extracts bond density, fits it with basis functions at the bond center,
    and symmetrizes the representation from both atom perspectives.

    Args:
        i0 (int): Index of first atom.
        i1 (int): Index of second atom.
        L (Lowdin_split): Löwdin-split density matrix object.
        mybasis (dict): Bond basis sets keyed by bond names.
        idx (dict): Pair indices for symmetrization.
        q (list): Element symbols for all atoms.
        r (numpy ndarray): Atomic coordinates in Å.
        cutoff (float): Maximum bond distance.

    Returns:
        tuple: ([v0, v1], bname) where:
        - v0: Representation from atom i0's perspective.
        - v1: Representation from atom i1's perspective.
        - bname: Bond name (e.g., 'CH').
        Returns (None, None) if distance exceeds cutoff.
    """
    q0, q1 = q[i0], q[i1]
    r0, r1 = r[i0], r[i1]
    z = r1-r0
    if np.linalg.norm(z) > cutoff:
        return None, None
    dm1   = L.get_bond(i0, i1)
    bname = make_bname(q0, q1)
    cs    = fit_dm(dm1, L.mol, mybasis[bname], r0, r1)
    lmax  = max(c[0] for c in cs)
    v0    = vec_from_cs(+z, cs, lmax, idx[bname])
    v1    = vec_from_cs(-z, cs, lmax, idx[bname])
    return [v0, v1], bname


def repr_for_mol(mol, dm, qqs, M, mybasis, idx, maxlen, cutoff, only_z=None):
    """Compute SPAHM(b) representations for all atoms in a molecule.

    Constructs bond-based atomic representations by summing contributions from
    all bonds of the same type within cutoff distance.

    Args:
        mol (pyscf.gto.Mole): Molecule object.
        dm (numpy.ndarray): Molecular density matrix.
        qqs (dict): Maps each element to list of bond pair names it can form.
        M (dict): Metric matrices for each bond type.
        mybasis (dict): Bond basis sets keyed by bond names.
        idx (dict): Pair indices for symmetrization of each bond type.
        maxlen (int): Maximum representation length for zero-padding.
        cutoff (float): Maximum bond distance in Å to consider.
        only_z (list, optional): If provided, compute representations only for atoms
            with these element symbols. Defaults to None (all atoms).

    Returns:
        numpy.ndarray: Array of shape (n_atoms, maxlen) containing atom representations,
            where each row is a zero-padded SPAHM(b) vector.
    """
    if only_z is None:
        only_z = []

    L = lowdin.Lowdin_split(mol, dm)
    q = [mol.atom_symbol(i) for i in range(mol.natm)]
    r = mol.atom_coords(unit='ANG')

    mybonds = [bonds_dict_init(qqs[q0], M) for q0 in q]
    if len(only_z) > 0:
        all_atoms = [ai for ai in range(mol.natm) if (mol.atom_pure_symbol(ai) in only_z)]
    else:
        all_atoms = range(mol.natm)
    for i0 in all_atoms:
        rest = mol.natm if len(only_z) > 0 else i0
        for i1 in range(rest):
            if i0 == i1 :
                continue
            v, bname = repr_for_bond(i0, i1, L, mybasis, idx, q, r, cutoff)
            if v is None:
                continue
            mybonds[i0][0][bname] += v[0]
            mybonds[i1][0][bname] += v[1]
    vec = [None]*len(all_atoms)
    for i1,i0 in enumerate(all_atoms):
        vec[i1] = np.hstack([M[qq] @ mybonds[i0][0][qq] for qq in qqs[q[i0]]])
        vec[i1] = np.pad(vec[i1], (0, maxlen-len(vec[i1])), 'constant')
    return np.array(vec)
