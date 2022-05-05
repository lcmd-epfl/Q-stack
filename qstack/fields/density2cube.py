import numpy as np
from pyscf.dft.numint import eval_ao
from pyscf.tools.cubegen import Cube

def coeffs_to_cube(mol, coeffs, cubename, nx = 80, ny = 80, nz = 80, resolution = 0.1, margin = 3.0):

    # Make grid
    grid = Cube(mol, nx, ny, nz, resolution, margin)

    # Compute density on the .cube grid
    coords = grid.get_coords()
    ngrids = grid.get_ngrids()
    
    ao = eval_ao(mol, coords)
    orb_on_grid = np.dot(ao, coeffs)
    orb_on_grid = orb_on_grid.reshape(grid.nx,grid.ny,grid.nz)

    # Write out orbital to the .cube file
    grid.write(orb_on_grid, cubename, comment='Electron Density')
