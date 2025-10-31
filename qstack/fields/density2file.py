import numpy as np
from pyscf.dft.numint import eval_ao
from pyscf.tools.cubegen import Cube
import pyscf.tools.molden
from .decomposition import number_of_electrons_deco

def coeffs_to_cube(mol, coeffs, cubename, nx = 80, ny = 80, nz = 80, resolution = 0.1, margin = 3.0):
    """Saves the electron density to a cube file format.

    Evaluates the density from expansion coefficients on a 3D grid and writes
    it to a Gaussian cube file for visualization.

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

    # Make grid
    grid = Cube(mol, nx, ny, nz, resolution, margin)

    # Compute density on the .cube grid
    coords = grid.get_coords()

    ao = eval_ao(mol, coords)
    orb_on_grid = np.dot(ao, coeffs)
    orb_on_grid = orb_on_grid.reshape(grid.nx,grid.ny,grid.nz)

    # Write out orbital to the .cube file
    grid.write(orb_on_grid, cubename, comment='Electron Density')


def coeffs_to_molden(mol, coeffs, moldenname):
    """Saves the electron density to a MOLDEN file format.

    Writes the density represented by expansion coefficients to a MOLDEN file
    which can be visualized with various quantum chemistry visualization tools.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        coeffs (numpy ndarray): 1D array of density expansion coefficients.
        moldenname (str): Output filename for the MOLDEN file.

    Returns:
        None: Creates a file named <moldenname>.molden on disk.
    """

    with open(moldenname, 'w') as f:
        N = number_of_electrons_deco(mol, coeffs)
        pyscf.tools.molden.header(mol, f, True)
        pyscf.tools.molden.orbital_coeff(mol, f, np.array([coeffs]).T, ene=[0.0], occ=[N], ignore_h=True)
