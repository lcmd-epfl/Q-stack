import numpy
import pyscf
from qstack import compound, fields


def spherical_atoms(elements, atm_bas):
    """Get density matrices for spherical atoms.

    Args:
        elements (list of str): elements to compute the DM for.
        atm_bas (string / pyscf basis dictionary): basis to use.

    Returns:
        dict of numpy 2d ndarrays: atomic density matrices for each element with its name as a key.

    """
    dm_atoms = {}
    for q in elements:
        mol_atm = pyscf.gto.M(atom=[[q, [0,0,0]]], spin=pyscf.data.elements.ELEMENTS_PROTON[q]%2, basis=atm_bas)
        dm_atoms[q] = pyscf.scf.hf.init_guess_by_atom(mol_atm)
    return dm_atoms


def _hirshfeld_weights(mol_full, grid_coord, atm_dm, atm_bas, dominant):

    # promolecular density
    grid_n = len(grid_coord)
    rho_atm = numpy.zeros((mol_full.natm, grid_n), dtype=float)
    for i in range(mol_full.natm):
        q = mol_full._atom[i][0]
        mol_atm    = pyscf.gto.M(atom=mol_full._atom[i:i+1], basis=atm_bas, spin=pyscf.data.elements.ELEMENTS_PROTON[q]%2, unit='Bohr')
        ao_atm     = pyscf.dft.numint.eval_ao(mol_atm, grid_coord)
        rho_atm[i] = pyscf.dft.numint.eval_rho(mol_atm, ao_atm, atm_dm[q])

    # get hirshfeld weights
    rho = rho_atm.sum(axis=0)
    idx = numpy.where(rho > 0)[0]
    h_weights = numpy.zeros_like(rho_atm)
    for i in range(mol_full.natm):
        h_weights[i,idx] = rho_atm[i,idx] /rho[idx]

    if dominant:
        # get dominant hirshfeld weights
        for point in range(grid_n):
            i = numpy.argmax(h_weights[:,point])
            h_weights[:,point] = numpy.zeros(mol_full.natm)
            h_weights[i,point] = 1.0
    return h_weights


def hirshfeld_charges(mol, cd, dm_atoms=None, atm_bas=None,
                      dominant=True,
                      occupations=False, grid_level=3):

    """Fit molecular density onto an atom-centered basis.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        cd (numpy 1d or 2d ndarray or list of arrays): density-fitting coefficients / density matrices.
        dm_atoms (dict of numpy 2d ndarrays): atomic density matrices (output of the `spherical_atoms` fn).
                                              If None, is computed on-the-fly.
        atm_bas (string / pyscf basis dictionary): basis set used to compute dm_atoms.
                                                   If None, is taken from mol.
        dominant (bool): whether to use dominant or classical partitioning.
        occupations (bool): whether to return atomic occupations or charges.
        grid level (int): grid level for numerical integration.

    Returns:
        numpy 1d ndarray or list of them: computed atomic charges or occupations.

    """

    def atom_contributions(cd, ao, tot_weights):
        if cd.ndim==1:
            tmp = numpy.einsum('i,xi->x', cd, ao)
        elif cd.ndim==2:
            tmp = numpy.einsum('pq,xp,xq->x', cd, ao, ao)
        return numpy.einsum('x,ax->a', tmp, tot_weights)

    # check input
    if type(cd)==list:
        cd_list = cd
    else:
        cd_list = [cd]

    # spherical atoms
    if atm_bas==None:
        atm_bas = mol.basis
    if dm_atoms==None:
        dm_atoms = spherical_atoms(set(mol.elements), atm_bas)

    # construct integration grid
    g = fields.dm.make_grid_for_rho(mol, grid_level=grid_level)

    # compute weights
    h_weights   = _hirshfeld_weights(mol, g.coords, dm_atoms, atm_bas, dominant)
    tot_weights = numpy.einsum('x,ax->ax', g.weights, h_weights)

    # atom partitioning
    ao  = pyscf.dft.numint.eval_ao(mol, g.coords)
    charges_list = [atom_contributions(i, ao, tot_weights) for i in cd_list]
    if not occupations:
        charges_list = [mol.atom_charges()-charges for charges in charges_list]

    if type(cd)==list:
        return charges_list
    else:
        return charges_list[0]
