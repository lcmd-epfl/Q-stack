from pyscf import dft, scf
from qstack import constants
import numpy as np

def get_converged_dm(mol, xc):
    """Performs restricted SCF and returns density matrix, given pyscf mol object and an XC density functional.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        xc (str): XC functional for computation.

    Returns:
        numpy ndarray: density matrix in AO-basis.

    """

    if mol.multiplicity == 1:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc
    print("Starting Kohn-Sham computation at "+str(mf.xc)+"/"+str(mol.basis)+" level.")
    mf.verbose = 1
    mf.kernel()

    print("Convergence: ",mf.converged)
    print("Energy: ",mf.e_tot)

    # Make the one-particle density matrix in ao-basis
    dm = mf.make_rdm1()

    return dm

def make_grid_for_rho(mol, grid_level = 3):
    """ Generates a grid of real space coordinates and weights for integration.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        grid_level (int): controls the number of radial and angular points.

    Returns:
        object : pyscf DFT mesh grid object

    """
    grid = dft.gen_grid.Grids(mol)
    grid.level = grid_level
    grid.build()

    return grid

def sphericalize_density_matrix(mol, dm):
    """Sphericalize the density matrix in the sense of an integral over all possible rotations.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        dm (numpy ndarray): density matrix in AO-basis.

    Returns:
        numpy ndarray: sphericalized density matrix.

    """

    idx_by_l = [[] for i in range(constants.MAX_L)]
    i0 = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        i1 = i0 + nc * (l*2+1)
        idx_by_l[l].extend(range(i0, i1, l*2+1))
        i0 = i1

    spherical_dm = np.zeros_like(dm)

    for l in np.nonzero(idx_by_l)[0]:
        for idx in idx_by_l[l]:
            for jdx in idx_by_l[l]:
                if l == 0:
                    spherical_dm[idx,jdx] = dm[idx,jdx]
                else:
                    trace = 0
                    for m in range(2*l+1):
                        trace += dm[idx+m,jdx+m]
                    for m in range(2*l+1):
                        spherical_dm[idx+m,jdx+m] = trace / (2*l+1)

    return spherical_dm

# TODO merge with get_converged_dm()
def get_converged_mf(mol, func, dm0=None):
    mf = scf.RKS(mol)
    mf.xc = func
    mf.kernel(dm0=dm0)
    dm = mf.make_rdm1()
    return (mf, dm)

