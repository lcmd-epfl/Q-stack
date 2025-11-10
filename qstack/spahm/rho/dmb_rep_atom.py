"""Functions for SPAHM(a) computation.

Implements various models: pure, SAD-diff, occupation-corrected, Löwdin partitioning.

Provides:
    models_dict: Dictionary of available models.
"""

import numpy as np
import pyscf
from qstack import compound, fields
from . import sym, atomic_density, lowdin
from qstack.tools import slice_generator


def get_basis_info(atom_types, auxbasis):
    """Gather auxiliary basis information for all atom types.

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
    """Create a dictionary of available SPAHM(a) models.

    Defines density fitting functions for each model.

    Returns:
        dict: Mapping model names to (density_fitting_function, symmetrization_function, maxlen_function).
    """
    def df_pure(mol, dm, auxbasis, only_i):
        """Pure density fitting without modifications."""
        auxmol, c = fields.decomposition.decompose(mol, dm, auxbasis)
        return sym.c_split_atom(auxmol, c, only_i=only_i)

    def df_sad_diff(mol, dm, auxbasis, only_i=None):
        """Density fitting on difference from superposition of atomic densities (SAD)."""
        mf = pyscf.scf.RHF(mol)
        dm_sad = mf.init_guess_by_atom(mol)
        if dm_sad.ndim==3:
            dm_sad = dm_sad.sum(axis=0)
        dm = dm - dm_sad
        auxmol, c = fields.decomposition.decompose(mol, dm, auxbasis)
        return sym.c_split_atom(auxmol, c, only_i=only_i)

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

    def df_occup(mol, dm, auxbasis, only_i=None):
        """Pure density fitting with preserving atom charges."""
        L = lowdin.Lowdin_split(mol, dm)
        diag = np.diag(L.dmL)
        Q = np.array([sum(diag[start:stop]) for (start, stop) in mol.aoslice_nr_by_atom()[:,2:]])
        auxmol = compound.make_auxmol(mol, auxbasis)
        eri2c, eri3c = fields.decomposition.get_integrals(mol, auxmol)[1:]
        c0 = fields.decomposition.get_coeff(dm, eri2c, eri3c)
        c  = fields.decomposition.correct_N_atomic(auxmol, Q, c0, metric=eri2c)
        return sym.c_split_atom(auxmol, c, only_i=only_i)


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
    """Return density fitting and symmetrization functions for specified model.

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
                only_i (list[int]): List of atom indices to use.

            Returns:
                list: Density fitting coefficients per atom (1D numpy ndarrays).

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


def coefficients_symmetrize_MR2021(maxlen, c, atoms, idx, ao, _, _M, only_i):
    """Symmetrize density fitting coefficients using MR2021 method.

    Reference:
        J. T. Margraf, K. Reuter,
        "Pure non-local machine-learned density functional theory for electron correlation",
        Nat. Commun. 12, 344 (2021), doi:10.1038/s41467-020-20471-y.

    Args:
        maxlen (int): Maximum feature length.
        c (list): List of coefficient arrays per atom.
        atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
        idx (dict): Pair indices per element.
        ao (dict): Angular momentum info per element.
        _: Unused (for interface compatibility).
        _M: Unused (for interface compatibility).
        only_i (list[int]): List of atom indices to use.

    Returns:
        numpy ndarray: 2D array (n_atoms, max_features) with zero-padding.
    """
    if only_i is not None and len(only_i)>0:
        atoms = np.array(atoms)[only_i]
    v = np.zeros((len(atoms), maxlen))
    for iat, (q, ci) in enumerate(zip(atoms, c, strict=True)):
        vi = sym.vectorize_c_MR2021(idx[q], ao[q], ci)
        v[iat,:len(vi)] = vi
    return v


def coefficients_symmetrize_short(maxlen, c, atoms, idx, ao, _, M, only_i):
    """Symmetrize coefficients for each atom.

    For each atom, use contributions from the said atom.

    Args:
        maxlen (int): Maximum feature length.
        c (list): List of coefficient arrays per atom.
        atoms (list[str]): Atoms in molecule (from pyscf Mole.elements).
        idx (dict): Pair indices per element.
        ao (dict): Angular momentum info per element.
        _: Unused (for interface compatibility).
        M (dict): Metric matrices per element.
        only_i (list[int]): List of atom indices to use.

    Returns:
        numpy ndarray: 2D array (n_atoms, max_features) with zero-padding.
    """
    if only_i is not None and len(only_i)>0:
        atoms = np.array(atoms)[only_i]
    v = np.zeros((len(atoms), maxlen))
    for iat, (q, ci) in enumerate(zip(atoms, c, strict=True)):
        v[iat,:len(idx[q])] = M[q] @ sym.vectorize_c_short(idx[q], ao[q], ci)
    return v


def coefficients_symmetrize_long(maxlen, c_df, atoms, idx, ao, ao_len, M, _):
    """Symmetrize coefficients for long Löwdin models.

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
