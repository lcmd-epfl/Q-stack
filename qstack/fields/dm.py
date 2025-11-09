"""Density matrix manipulation and analysis functions."""

import numpy as np
from pyscf import dft
from qstack import constants
from qstack.tools import Cursor


def get_converged_mf(mol, xc, dm0=None, verbose=False):
    """Perform SCF calculation.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        xc (str): Exchange-correlation functional.
        dm0 (numpy ndarray, optional): Initial guess for density matrix. Defaults to None.
        verbose (bool): If print more information.

    Returns:
        tuple: A tuple containing:
        - mf (pyscf.dft.rks.RKS or pyscf.dft.uks.UKS): Converged mean-field object.
        - dm (numpy ndarray): Converged density matrix in AO-basis.
    """
    if mol.multiplicity == 1:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc
    if verbose:
        print(f"Starting Kohn-Sham computation at {mf.xc}/{mol.basis} level.")
    mf.verbose = 1
    mf.kernel(dm0=dm0)

    if verbose:
        print(f"Convergence: {mf.converged}")
        print(f"Energy: {mf.e_tot}")

    dm = mf.make_rdm1()
    return (mf, dm)


def get_converged_dm(mol, xc, verbose=False):
    """Get a converged density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        xc (str): Exchange-correlation functional.
        verbose (bool): If print more information.

    Returns:
        A numpy ndarray containing the density matrix in AO-basis.
    """
    return get_converged_mf(mol, xc, dm0=None, verbose=verbose)[1]


def make_grid_for_rho(mol, grid_level=3):
    """Generate a grid of real space coordinates and weights for integration.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        grid_level (int): Controls the number of radial and angular points.

    Returns:
        pyscf Grid object.
    """
    grid = dft.gen_grid.Grids(mol)
    grid.level = grid_level
    grid.build()
    return grid


def sphericalize_density_matrix(mol, dm):
    """Sphericalize the density matrix in the sense of an integral over all possible rotations.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (2D numpy array): Density matrix in AO-basis.

    Returns:
        A numpy ndarray with the sphericalized density matrix.
    """
    idx_by_l = [[] for i in range(constants.MAX_L)]
    i0 = Cursor(action='ranger')
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        msize = 2*l+1
        nc = mol.bas_nctr(ib)
        idx_by_l[l].extend(i0(nc*msize)[::msize])

    spherical_dm = np.zeros_like(dm)

    for l in np.nonzero(idx_by_l)[0]:
        msize = 2*l+1
        for idx in idx_by_l[l]:
            for jdx in idx_by_l[l]:
                if l == 0:
                    spherical_dm[idx,jdx] = dm[idx,jdx]
                else:
                    trace = 0
                    for m in range(msize):
                        trace += dm[idx+m,jdx+m]
                    for m in range(msize):
                        spherical_dm[idx+m,jdx+m] = trace / msize

    return spherical_dm

