import numpy
from pyscf.dft.numint import eval_ao
from pyscf.tools.cubegen import Cube
import pyscf.tools.molden
from qstack.fields.decomposition import number_of_electrons_deco

def coeffs_to_cube(mol, coeffs, cubename, nx = 80, ny = 80, nz = 80, resolution = 0.1, margin = 3.0):

    # Make grid
    grid = Cube(mol, nx, ny, nz, resolution, margin)

    # Compute density on the .cube grid
    coords = grid.get_coords()
    ngrids = grid.get_ngrids()

    ao = eval_ao(mol, coords)
    orb_on_grid = numpy.dot(ao, coeffs)
    orb_on_grid = orb_on_grid.reshape(grid.nx,grid.ny,grid.nz)

    # Write out orbital to the .cube file
    grid.write(orb_on_grid, cubename, comment='Electron Density')


def coeffs_to_molden(mol, coeffs, moldenname):
    with open(moldenname, 'w') as f:
        pyscf.tools.molden.header(mol, f, True)
        try:
            N = number_of_electrons_deco(mol, coeffs)
        except:
            N = 0.0
        pyscf.tools.molden.orbital_coeff(mol, f, numpy.array([coeffs]).T, ene=[0.0], occ=[N], ignore_h=True)

