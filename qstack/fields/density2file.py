import numpy as np
from pyscf.dft.numint import eval_ao
from pyscf.tools.cubegen import Cube
import pyscf.tools.molden
from . import moments

def coeffs_to_cube(mol, coeffs, cubename, nx=80, ny=80, nz=80, resolution=0.1, margin=3.0):
    """Saves the electron density to a cube file.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        coeffs (numpy ndarray): 1D array of density expansion coefficients.
        cubename (str): Output filename (without .cube extension).
        nx (int): Number of grid points in x direction. Defaults to 80.
        ny (int): Number of grid points in y direction. Defaults to 80.
        nz (int): Number of grid points in z direction. Defaults to 80.
        resolution (float): Grid spacing in Bohr. Defaults to 0.1.
        margin (float): Extra space around molecule in Bohr. Defaults to 3.0.

    Returns:
        None: Creates a file named <cubename>.cube on disk.
    """

    grid = Cube(mol, nx, ny, nz, resolution, margin)
    coords = grid.get_coords()
    ao = eval_ao(mol, coords)
    orb_on_grid = np.dot(ao, coeffs)
    orb_on_grid = orb_on_grid.reshape(grid.nx, grid.ny, grid.nz)
    grid.write(orb_on_grid, cubename, comment='Electron Density')


def coeffs_to_molden(mol, coeffs, moldenname):
    """Saves the electron density to a MOLDEN file.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        coeffs (numpy ndarray): 1D array of density expansion coefficients.
        moldenname (str): Output filename for the MOLDEN file.

    Returns:
        None: Creates a file named <moldenname>.molden on disk.
    """

    with open(moldenname, 'w') as f:
        N = moments.r2_c(mol, coeffs, moments=[0])[0]
        pyscf.tools.molden.header(mol, f, True)
        pyscf.tools.molden.orbital_coeff(mol, f, np.array([coeffs]).T, ene=[0.0], occ=[N], ignore_h=True)
