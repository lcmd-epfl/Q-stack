#!/usr/bin/env python3

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import KernelCenterer
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from qstack.qml import b2r2, slatm

def _restore_from_pickle(objname: str, version: int, hypers: dict, params: dict|None):
    pass






class B2R2Representation(TransformerMixin, BaseEstimator):
    """Transform reactions into their B2R2 representations

    Reference:
        P. van Gerwen, A. Fabrizio, M. D. Wodrich, C. Corminboeuf,
        "Physics-based representations for machine learning properties of chemical reactions",
        Mach. Learn.: Sci. Technol. 3, 045005 (2022), doi:10.1088/2632-2153/ac8f1a

    This representation can be computed for molecules or for reactions,
    by giving to this transformer a list of one or a list of the other.
    Note that no fitting is required, and this object is a simple shim best used
    in pipeline objects.

    Molecules are ASE molecule objects, or any object with `.numbers` and `.positions` (in Å) properties.
    Reactions, however, are ``qstack.qml.b2r2.Reaction objects,
    or tuples of two lists of molecules (reactants, products).

    Parameters
    ----------
    variant: str, default "l"
        B2R2 variant to compute. Options:
            - 'l': Local variant with element-resolved skewed Gaussians (default).
            - 'a': Agnostic variant with element-pair Gaussians.
            - 'n': Nuclear variant with combined skewed Gaussians.

    progress: bool, default False
        If True, displays progress bar

    rcut: float, default 3.5
        Cutoff radius for bond detection in Å

    gridspace: float, default 0.03
        Grid spacing for discretization in Å


    Attributes
    ----------
    None

    Examples
    --------
    [ fixme ]
    """

    def __init__(
        self,
        variant='l',
        progress=False,
        rcut=b2r2.defaults.rcut,
        gridspace=b2r2.defaults.gridspace,
    ):
        """Initialize StandardFlexibleScaler."""
        self.variant = variant
        self.progress = progress
        self.rcut = rcut
        self.gridspace = gridspace
        self.elements_ = []

    def __reduce__(self):
        return (
            _restore_from_pickle,
            "B2R2", 1,
            dict(
                variant = self.variant,
                progress = self.progress,
                rcut = self.rcut,
                gridspace = self.gridspace,
            ),
            {'elements_': self.elements} if self.elements else None,
        )

    def fit(self, X, y=None, sample_weight=None):
        """Determine the types of elements to consider, by feeding them from all objects to consider later.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y: None
            Ignored.
        sample_weight: numpy.ndarray of shape (n_samples,)
            Weights for each sample. Sample weighting can be used to center
            (and scale) data using a weighted mean. Weights are internally
            normalized before preprocessing.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        elems = set()
        for obj in X:
            if isinstance(obj, tuple) and len(obj) == 2:
                reac_mols = obj[0] + obj[1]
            elif isinstance(obj, b2r2.Reaction):
                reac_mols = obj.reactants + obj.products
            elif hasattr(X[0], "numbers") and hasattr(X[0], "positions"):
                reac_mols = [obj]
            for mol in reac_mols:
                elems.update(mol.numbers)
        self.elements_ = sorted(elems)

        return self

    def transform(self, X, y=None, copy=None):
        """Normalize a vector based on previously computed mean and scaling.

        Parameters
        ----------
        X : list of length n_samples, of molecules or list of reactions
            The chemical objects to compute representations of.
            Please note they should all be of the same type (reactions OR molecules)
        y: None
            Ignored.
        copy : bool, default=None
            Ignored

        Returns
        -------
        X : {array-like} of shape (n_samples, n_features)
            Transformed data.
        """

        if self.variant=='l':
            get_b2r2_molecular=b2r2.get_b2r2_l_molecular
            combine = lambda r,p: p-r
        elif self.variant=='a':
            get_b2r2_molecular = b2r2.get_b2r2_a_molecular
            combine = lambda r,p: p-r
        elif self.variant=='n':
            get_b2r2_molecular=b2r2.get_b2r2_n_molecular
            combine = lambda r,p: np.hstack((r,p))
        else:
            raise RuntimeError(f'Unknown B2R2 {variant=}')

        if isinstance(X[0], tuple) and len(X[0]) == 2:
            mode = "reac-2"
            first_array = self._get_reac_array(X[0][0], X[0][1], get_b2r2_molecular, combine)
        elif isinstance(X[0], b2r2.Reaction):
            mode = "reac"
            first_array = self._get_reac_array(X[0].reactants, X[0].products, get_b2r2_molecular, combine)
        elif hasattr(X[0], "numbers") and hasattr(X[0], "positions"):
            mode = "mol"
            first_array = self._get_mol_array(X[0], get_b2r2_molecular)
        else:
            raise ValueError("unknown type of input")

        assert first_array.ndim==1
        full_array = np.empty_like(first_array, shape=(len(X), *first_array.shape))
        full_array[0] = first_array

        for object_i,x in enumerate(X[1:]):
            if mode == "reac-2":
                full_array[object_i+1] = self._get_reac_array(x[0], x[1], get_b2r2_molecular, combine)
            elif mode == "reac":
                full_array[object_i+1] = self._get_reac_array(x.reactants, x.products, get_b2r2_molecular, combine)
            elif mode == "mol":
                full_array[object_i+1] = self._get_mol_array(x, get_b2r2_molecular)
        return full_array

    def _get_reac_array(self, reactants, products, mol_rep_func, combine):
        reac_repr = self._get_mol_array(reactants[0], mol_rep_func)
        for reac in reactants[1:]:
            reac_repr += self._get_mol_array(reac, mol_rep_func)
        prod_repr = self._get_mol_array(products[0], mol_rep_func)
        for prod in products[1:]:
            prod_repr += self._get_mol_array(prod, mol_rep_func)
        return combine(reac_repr, prod_repr)

    def _get_mol_array(self, mol, mol_rep_func):
        return mol_rep_func(mol.numbers, mol.positions, self.rcut, self.gridspace, self.elements_)
