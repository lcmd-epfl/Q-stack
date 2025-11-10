"""Löwdin orthogonalization for density matrix partitioning."""

import numpy as np


class Lowdin_split:
    """Löwdin orthogonalization for density matrix partitioning.

    Transforms density matrix to orthogonal basis using symmetric orthogonalization,
    enabling clean atomic and bond partitioning of electron density.

    Attributes:
        S (numpy ndarray): Overlap matrix in AO basis.
        S12 (numpy ndarray): Square root of overlap matrix (S^{1/2}).
        S12i (numpy ndarray): Inverse square root of overlap matrix (S^{-1/2}).
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Original density matrix in AO basis.
        dmL (numpy ndarray): Löwdin-orthogonalized density matrix.
    """
    def __init__(self, mol, dm):
        """Initialize Löwdin split with molecule and density matrix.

        Args:
            mol (pyscf Mole): pyscf Mole object.
            dm (numpy ndarray): Density matrix in AO basis.
        """
        S = mol.intor_symmetric('int1e_ovlp')
        S12,S12i = self.sqrtm(S)
        self.S    = S
        self.S12  = S12
        self.S12i = S12i
        self.mol  = mol
        self.dm   = dm
        self.dmL  = S12 @ dm @ S12

    def sqrtm(self, m):
        """Compute matrix square root and inverse square root via eigendecomposition.

        Args:
            m (numpy ndarray): Symmetric positive-definite matrix.

        Returns:
            tuple: (m^{1/2}, m^{-1/2}) both symmetrized.
        """
        e,b = np.linalg.eigh(m)
        e   = np.sqrt(e)
        sm  = b @ np.diag(e    ) @ b.T
        sm1 = b @ np.diag(1.0/e) @ b.T
        return (sm+sm.T)*0.5, (sm1+sm1.T)*0.5

    def get_bond(self, at1idx, at2idx):
        """Extract bond density matrix for an atom pair.

        Isolates the density matrix components corresponding to interactions
        between two atoms, transforming back to AO basis.

        Args:
            at1idx (int): Index of first atom.
            at2idx (int): Index of second atom.

        Returns:
            numpy ndarray: Bond density matrix in AO basis (2D array).
        """
        mo1idx = range(*self.mol.aoslice_nr_by_atom()[at1idx][2:])
        mo2idx = range(*self.mol.aoslice_nr_by_atom()[at2idx][2:])
        ix1 = np.ix_(mo1idx,mo2idx)
        ix2 = np.ix_(mo2idx,mo1idx)
        dmL_bond = np.zeros_like(self.dmL)
        dmL_bond[ix1] = self.dmL[ix1]
        dmL_bond[ix2] = self.dmL[ix2]
        return self.S12i @ dmL_bond @ self.S12i
