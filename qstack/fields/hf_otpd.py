from pyscf import dft
import numpy as np
from .dm import make_grid_for_rho


def hf_otpd(mol, dm, grid_level=3, save_otpd=False, return_all=False):
    """Computes the Hartree-Fock uncorrelated on-top pair density (OTPD) on a grid.

    The on-top pair density is the probability density of finding two electrons
    at the same position. For Hartree-Fock, this is computed as (rho/2)^2.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): 2D density matrix in AO-basis.
        grid_level (int): DFT grid level controlling number of radial and angular points. Defaults to 3.
        save_otpd (bool): If True, saves results to a .npz file. Defaults to False.
        return_all (bool): If True, returns both OTPD and grid object; if False, returns only OTPD. Defaults to False.

    Returns:
        numpy ndarray or tuple: If return_all is False, returns 1D array of OTPD values.
            If return_all is True, returns tuple of (otpd, grid) where grid is the pyscf Grid object.
    """

    grid = make_grid_for_rho(mol, grid_level)

    ao = dft.numint.eval_ao(mol, grid.coords)
    rho = np.einsum('pq,ip,iq->i', dm, ao, ao)

    hf_otpd = np.power(rho / 2, 2)

    if save_otpd:
        save_OTPD(mol, hf_otpd, grid)

    if return_all:
        return hf_otpd, grid
    return hf_otpd


def save_OTPD(mol, otpd, grid):
    """Saves on-top pair density computation results to a NumPy compressed file.

    Creates a .npz file containing the molecular structure, OTPD values,
    grid coordinates, and integration weights for later analysis.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        otpd (numpy ndarray): 1D array of on-top pair density values on the grid.
        grid (pyscf Grid): Grid object containing coordinates and weights.

    Returns:
        None: Creates a file named <elements>_otpd_data.npz on disk.
    """

    output = ''.join(mol.elements)+"_otpd_data"
    np.savez(output, atom=mol.atom, rho=otpd, coords=grid.coords, weights=grid.weights)
