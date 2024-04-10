from pyscf import dft
import numpy as np
from qstack.fields.dm import make_grid_for_rho

def hf_otpd(mol, dm, grid_level = 3, save_otpd = False, return_all = False):
    """Computes the uncorrelated on-top pair density on a grid.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): Density matrix in AO-basis.
        grid_level (int): Controls the number of radial and angular points.
        save_otpd (bool): If True, saves the input and output in a .npz file. Defaults to False
        return_all (bool): If true, returns the uncorrelated on-top pair density on a grid, and the cooresponding pyscf Grid object; if False, returns only the uncorrelated on-top pair density. Defaults to False

    Returns:
        A numpy ndarray with the uncorrelated on-top pair density on a grid. If 'return_all' = True, then it also returns the corresponding pyscf Grid object.
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
    """ Saves the information about an OTPD computation into a .npz file.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        otpd (numpy ndarray): On-top pair density on a grid.
        grid (pyscf Grid): Grid object
    """

    output = ''.join(mol.elements)+"_otpd_data"
    np.savez(output, atom=mol.atom, rho=otpd, coords=grid.coords, weights=grid.weights)
