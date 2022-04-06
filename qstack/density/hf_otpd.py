from pyscf import dft
import numpy as np
from qstack.density.dm import make_grid_for_rho

def hf_otpd(mol, dm, grid_level = 3, save_otpd = False):
    """ Computes the uncorrelated on-top pair density on a grid.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): density matrix in AO-basis.
        grid_level (int): controls the number of radial and angular points.

    Returns:
        numpy ndarray : uncorrelated on-top pair density on a grid.
    
    """

    grid = make_grid_for_rho(mol, grid_level)

    ao = dft.numint.eval_ao(mol, grid.coords)
    rho = np.einsum('pq,ip,iq->i', dm, ao, ao)

    hf_otpd = np.power(rho / 2, 2)

    if save_otpd:
        save_OTPD(mol, hf_otpd, grid)

    return hf_otpd

def save_OTPD(mol, otpd, grid):
    """ Saves the information about an OTPD computation into a single file in uncompressed .npz format.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        otpd (numpy ndarray): on-top pair density on a grid.
        grid (pyscf Grid): Grid object
    """

    output = ''.join(mol.elements)+"_otpd_data"
    np.savez(output, atom=mol.atom, rho=otpd, coords=grid.coords, weights=grid.weights)