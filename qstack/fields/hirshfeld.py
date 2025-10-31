import numpy as np
import pyscf
from .dm import make_grid_for_rho


def spherical_atoms(elements, atm_bas):
    """Computes density matrices for spherically averaged isolated atoms.

    For each element, creates an isolated atom calculation with appropriate spin
    and computes its density matrix using atomic Hartree-Fock initial guess.

    Args:
        elements (list of str or set): Element symbols to compute density matrices for.
        atm_bas (str or dict): Basis set name (e.g., 'def2-svp') or pyscf basis dictionary.

    Returns:
        dict: Dictionary mapping element symbols (str) to atomic density matrices (numpy 2D ndarrays).
    """

    dm_atoms = {}
    for q in elements:
        mol_atm = pyscf.gto.M(atom=[[q, [0,0,0]]], spin=pyscf.data.elements.ELEMENTS_PROTON[q]%2, basis=atm_bas)
        dm_atoms[q] = pyscf.scf.hf.init_guess_by_atom(mol_atm)
    return dm_atoms

def _hirshfeld_weights(mol_full, grid_coord, atm_dm, atm_bas, dominant):
    """Computes Hirshfeld partitioning weights for each atom at grid points.

    Hirshfeld partitioning divides the molecular density among atoms based on
    their promolecular (free atom) densities. Dominant partitioning assigns
    each grid point exclusively to the atom with the highest weight.

    Args:
        mol_full (pyscf Mole): Complete molecular pyscf Mole object.
        grid_coord (numpy ndarray): 2D array (ngrids, 3) of grid point coordinates in Bohr.
        atm_dm (dict): Dictionary mapping element symbols to atomic density matrices from `spherical_atoms`.
        atm_bas (str or dict): Basis set name or dictionary used for atomic density matrices.
        dominant (bool): If True, uses dominant (all-or-nothing) partitioning; if False, uses standard Hirshfeld weights.

    Returns:
        numpy ndarray: 2D array (natm, ngrids) of partitioning weights for each atom at each grid point.
    """

    # promolecular density
    grid_n = len(grid_coord)
    rho_atm = np.zeros((mol_full.natm, grid_n), dtype=float)
    for i in range(mol_full.natm):
        q = mol_full._atom[i][0]
        mol_atm    = pyscf.gto.M(atom=mol_full._atom[i:i+1], basis=atm_bas, spin=pyscf.data.elements.ELEMENTS_PROTON[q]%2, unit='Bohr')
        ao_atm     = pyscf.dft.numint.eval_ao(mol_atm, grid_coord)
        rho_atm[i] = pyscf.dft.numint.eval_rho(mol_atm, ao_atm, atm_dm[q])

    # get hirshfeld weights
    rho = rho_atm.sum(axis=0)
    idx = np.where(rho > 0)[0]
    h_weights = np.zeros_like(rho_atm)
    for i in range(mol_full.natm):
        h_weights[i,idx] = rho_atm[i,idx] /rho[idx]

    if dominant:
        # get dominant hirshfeld weights
        for point in range(grid_n):
            i = np.argmax(h_weights[:,point])
            h_weights[:,point] = np.zeros(mol_full.natm)
            h_weights[i,point] = 1.0
    return h_weights


def hirshfeld_charges(mol, cd, dm_atoms=None, atm_bas=None,
                      dominant=True,
                      occupations=False, grid_level=3):
    """Computes atomic charges or occupations using Hirshfeld partitioning.

    Partitions the molecular electron density among atoms using Hirshfeld weights
    based on free atom densities. Can work with either density-fitting coefficients
    or full density matrices, and supports both standard and dominant partitioning.

    Args:
        mol (pyscf Mole): pyscf Mole object for the molecule.
        cd (numpy ndarray or list): Density representation as:
            - 1D array: density-fitting coefficients
            - 2D array: density matrix in AO basis
            - list: multiple densities (returns list of results)
        dm_atoms (dict, optional): Pre-computed atomic density matrices from `spherical_atoms`.
            If None, computed automatically. Defaults to None.
        atm_bas (str or dict, optional): Basis set for atomic density matrices.
            If None, uses mol.basis. Defaults to None.
        dominant (bool): If True, uses dominant (all-or-nothing) partitioning;
            if False, uses standard Hirshfeld weights. Defaults to True.
        occupations (bool): If True, returns atomic electron populations;
            if False, returns atomic charges (Z - N). Defaults to False.
        grid_level (int): DFT grid level for numerical integration. Defaults to 3.

    Returns:
        numpy ndarray or list: Atomic charges or occupations.
            - Single 1D array if cd is a single density
            - List of 1D arrays if cd is a list of densities
    """

    def atom_contributions(cd, ao, tot_weights):
        if cd.ndim==1:
            tmp = np.einsum('i,xi->x', cd, ao)
        elif cd.ndim==2:
            tmp = np.einsum('pq,xp,xq->x', cd, ao, ao)
        return np.einsum('x,ax->a', tmp, tot_weights)

    # check input
    if type(cd) is list:
        cd_list = cd
    else:
        cd_list = [cd]

    # spherical atoms
    if atm_bas is None:
        atm_bas = mol.basis
    if dm_atoms is None:
        dm_atoms = spherical_atoms(set(mol.elements), atm_bas)

    # construct integration grid
    g = make_grid_for_rho(mol, grid_level=grid_level)

    # compute weights
    h_weights   = _hirshfeld_weights(mol, g.coords, dm_atoms, atm_bas, dominant)
    tot_weights = np.einsum('x,ax->ax', g.weights, h_weights)

    # atom partitioning
    ao  = pyscf.dft.numint.eval_ao(mol, g.coords)
    charges_list = [atom_contributions(i, ao, tot_weights) for i in cd_list]
    if not occupations:
        charges_list = [mol.atom_charges()-charges for charges in charges_list]

    if type(cd) is list:
        return charges_list
    else:
        return charges_list[0]
