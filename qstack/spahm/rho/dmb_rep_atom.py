"""Functions for SPAHM(a) computation.

Implements various models: pure, SAD-diff, occupation-corrected, Löwdin partitioning.

Provides:
    - models_dict: Dictionary of available models.
"""

import numpy as np
import pyscf
from qstack import compound, fields
from . import sym, atomic_density, lowdin
from qstack.tools import slice_generator


def get_basis_info(atom_types, auxbasis):
    """Gathers auxiliary basis information for all atom types.

    Computes overlap matrices, basis function indices, and metric matrices
    needed for atomic density fitting.

    Args:
        atom_types (list): List of element symbols (e.g., ['C', 'H', 'O']).
        auxbasis (str or dict): Auxiliary basis set specification.

    Returns:
        tuple: (ao, ao_len, idx, M) where:
        - ao (dict): Angular momentum info per element.
        - ao_len (dict): Basis set size per element.
        - idx (dict): Pair indices for symmetrization per element.
        - M (dict): Metric matrices (2D numpy ndarray) per element.
    """
    ao = {}
    idx = {}
    M = {}
    ao_len = {}
    for q in atom_types:
        S, ao[q], ao_start = sym.get_S(q, auxbasis)
        idx[q] = sym.store_pair_indices_short(ao[q], ao_start)
        M[q]   = sym.metric_matrix_short(idx[q], ao[q], S)
        ao_len[q] = len(S)
    return ao, ao_len, idx, M


def _make_models_dict():
    """Creates dictionary of available SPAHM(a) models.

    Defines density fitting functions for each model.

    Returns:
        dict: Mapping model names to (density_fitting_function, symmetrization_function, maxlen_function).
    """
    def df_pure(mol, dm, auxbasis, **kwargs):
        """Pure density fitting without modifications."""
        return fields.decomposition.decompose(mol, dm, auxbasis)[1]

    def df_sad_diff(mol, dm, auxbasis, **kwargs):
        """Density fitting on difference from superposition of atomic densities (SAD)."""
        mf = pyscf.scf.RHF(mol)
        dm_sad = mf.init_guess_by_atom(mol)
        dm = dm - dm_sad
        return fields.decomposition.decompose(mol, dm, auxbasis)[1]

    def df_lowdin_long(mol, dm, auxbasis, only_i=None):
        """Löwdin partitioning with block-diagonal slicing with contributions from other elements."""
        return atomic_density.fit(mol, dm, auxbasis, only_i=only_i)

    def df_lowdin_short(mol, dm, auxbasis, only_i=None):
        """Löwdin partitioning with block-diagonal slicing."""
        return atomic_density.fit(mol, dm, auxbasis, short=True, only_i=only_i)

    def df_lowdin_long_x(mol, dm, auxbasis, only_i=None):
        """Löwdin partitioning with contributions from other elements."""
        return atomic_density.fit(mol, dm, auxbasis, w_slicing=False, only_i=only_i)

    def df_lowdin_short_x(mol, dm, auxbasis, only_i=None):
        """Löwdin partitioning."""
        return atomic_density.fit(mol, dm, auxbasis, short=True, w_slicing=False, only_i=only_i)

    def df_occup(mol, dm, auxbasis):
        """Pure density fitting with preserving atom charges."""
        L = lowdin.Lowdin_split(mol, dm)
        diag = np.diag(L.dmL)
        Q = np.array([sum(diag[start:stop]) for (start, stop) in mol.aoslice_nr_by_atom()[:,2:]])
        auxmol = compound.make_auxmol(mol, auxbasis)
        eri2c, eri3c = fields.decomposition.get_integrals(mol, auxmol)[1:]
        c0 = fields.decomposition.get_coeff(dm, eri2c, eri3c)
        c  = fields.decomposition.correct_N_atomic(auxmol, Q, c0, metric=eri2c)
        return c

    def maxlen_long(idx, _):
        return sum(len(v) for v in idx.values())

    def maxlen_short(idx, elements):
        return max(len(idx[q]) for q in elements)

    def maxlen_MR2021(idx, elements):
        return max(len(np.unique(idx[q][:,0])) for q in elements)

    models_dict = {'pure'          : (df_pure,            coefficients_symmetrize_short ,  maxlen_short  ),
                   'sad-diff'      : (df_sad_diff,        coefficients_symmetrize_short ,  maxlen_short  ),
                   'occup'         : (df_occup,           coefficients_symmetrize_short ,  maxlen_short  ),
                   'lowdin-short'  : (df_lowdin_short,    coefficients_symmetrize_short ,  maxlen_short  ),
                   'lowdin-long'   : (df_lowdin_long,     coefficients_symmetrize_long  ,  maxlen_long   ),
                   'lowdin-short-x': (df_lowdin_short_x,  coefficients_symmetrize_short ,  maxlen_short  ),
                   'lowdin-long-x' : (df_lowdin_long_x,   coefficients_symmetrize_long  ,  maxlen_long   ),
                   'mr2021'        : (df_pure,            coefficients_symmetrize_MR2021,  maxlen_MR2021 )}
    return models_dict


def get_model(arg):
    """Returns density fitting and symmetrization functions for specified model.

    Args:
        arg (str): Model name. Available options:
            - 'pure': Pure density fitting
            - 'occup': Occupation-corrected density fitting.
            - 'sad-diff': Superposition of Atomic Densities difference.
            - 'lowdin-short': Short Löwdin partitioning with slicing.
            - 'lowdin-long': Long Löwdin partitioning with slicing.
            - 'lowdin-short-x': Short Löwdin.
            - 'lowdin-long-x': Long Löwdin.
            - 'mr2021': Method from Margraf & Reuter 2021.

    Returns:
        tuple: (density_fitting_function, symmetrization_function, maxlen_function).
            - density_fitting_function (callable): Function performing density fitting.
            Args:
                mol (pyscf Mole): Molecule object.
                dm (numpy ndarray): Density matrix (2D).
                auxbasis (str or dict): Auxiliary basis set.
            Returns:
                numpy ndarray or list: Density fitting coefficients (1D).
            - symmetrization_function (callable): Function for symmetrizing coefficients.
            Args:
                maxlen (int): Maximum feature length.
                c (numpy ndarray): Density fitting coefficients (1D).
                atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
                idx (dict): Pair indices per element.
                ao (dict): Angular momentum info per element.
                ao_len (dict): Basis set sizes per element.
                M (dict): Metric matrices per element (2D numpy ndarrays).
                only_i (list[int]): List of atom indices to use.
            Returns:
                numpy ndarray: Symmetrized atomic feature vectors.
            - maxlen_function (callable): Function computing max. feature size.
            Args:
                idx (dict): Pair indices per element.
                elements (list[str]): Elements for which representation is computed.
            Returns:
                int: Maximum feature length.

    Raises:
        RuntimeError: If model name is not recognized.
    """
    arg = arg.lower()
    if arg not in models_dict:
        raise RuntimeError(f'Unknown model. Available models: {list(models_dict.keys())}')
    return models_dict[arg]


def coefficients_symmetrize_MR2021(maxlen, c, atoms, idx, ao, ao_len, _M, only_i):
    """Symmetrizes density fitting coefficients using MR2021 method.

    Reference:
        J. T. Margraf, K. Reuter,
        "Pure non-local machine-learned density functional theory for electron correlation",
        Nat. Commun. 12, 344 (2021), doi:10.1038/s41467-020-20471-y.

    Args:
        maxlen (int): Maximum feature length.
        c (numpy ndarray): Concatenated density fitting coefficients.
        atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
        idx (dict): Pair indices per element.
        ao (dict): Angular momentum info per element.
        ao_len (dict): Basis set sizes per element.
        _M: Unused (for interface compatibility).
        only_i (list[int]): List of atom indices to use.

    Returns:
        numpy ndarray: 2D array (n_atoms, max_features) with zero-padding.
    """
    if only_i is not None and len(only_i)>0:
        atoms = np.array(atoms)[only_i]
    v = np.zeros((len(atoms), maxlen))
    for iat, (q, ao_slice) in enumerate(slice_generator(atoms, inc=lambda q: ao_len[q])):
        vi = sym.vectorize_c_MR2021(idx[q], ao[q], c[ao_slice])
        v[iat,:len(vi)] = vi
    return v


def coefficients_symmetrize_short(maxlen, c, atoms, idx, ao, ao_len, M, only_i):
    """Symmetrizes coefficients for each atom.

    For each atom, use contributions from the said atom.

    Args:
        maxlen (int): Maximum feature length.
        c (numpy ndarray): Density fitting coefficients.
        atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
        idx (dict): Pair indices per element.
        ao (dict): Angular momentum info per element.
        ao_len (dict): Basis set sizes per element.
        M (dict): Metric matrices per element.
        only_i (list[int]): List of atom indices to use.

    Returns:
        numpy ndarray: 2D array (n_atoms, max_features) with zero-padding.
    """
    if only_i is not None and len(only_i)>0:
        atoms = np.array(atoms)[only_i]
    v = np.zeros((len(atoms), maxlen))
    for iat, (q, ao_slice) in enumerate(slice_generator(atoms, inc=lambda q: ao_len[q])):
        v[iat,:len(idx[q])] = M[q] @ sym.vectorize_c_short(idx[q], ao[q], c[ao_slice])
    return v


def coefficients_symmetrize_long(maxlen, c_df, atoms, idx, ao, ao_len, M, _):
    """Symmetrizes coefficients for long Löwdin models.

    For each atom, use contributions from the said atom as well as all other atoms.

    Args:
        maxlen (int): Maximum feature length.
        c_df (list): List of coefficient arrays per atom.
        atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
        idx (dict): Pair indices per element.
        ao (dict): Angular momentum info per element.
        ao_len (dict): Basis set sizes per element.
        M (dict): Metric matrices per element.
        _: Unused (for interface compatibility).

    Returns:
        numpy ndarray: 2D array (n_atoms, max_features) with zero-padding.
    """
    vectors = np.zeros((len(c_df), maxlen))
    feature_slice = dict(slice_generator(idx.keys(), inc=lambda q: len(idx[q])))
    for iat, c_a in enumerate(c_df):
        for q, ao_slice in slice_generator(atoms, inc=lambda q: ao_len[q]):
            vectors[iat,feature_slice[q]] += M[q] @ sym.vectorize_c_short(idx[q], ao[q], c_a[ao_slice])
    return vectors


models_dict = _make_models_dict()
