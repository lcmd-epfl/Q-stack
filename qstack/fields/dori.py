import numpy as np
from pyscf.dft.numint import eval_ao, _dot_ao_dm, _contract_rho
from qstack.fields.dm import make_grid_for_rho


def eval_rho(mol, ao, dm):
    r'''Calculate the electron density and the density derivatives.

    Taken from pyscf/dft/numint.py and modified to return second derivative matrices.

    Args:
        mol : an instance of :class:`Mole`
        ao : 3D array of shape (10,ngrids,nao).
            ao[0] is AO value and ao[1:3] are the AO gradients, ao[4:10] are AO second derivatives
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian)

    Returns:
        A tuple of:
            1D array of size ngrids to store electron density;
            2D array of (3,ngrids) to store density derivatives;
            3D array of (3,3,ngrids) to store 2nd derivatives
    '''

    ngrids, nao = ao.shape[-2:]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    AO = ao[0]
    dAO_dr = ao[1:4]
    d2AO_dr2 = np.zeros((3, 3, ngrids, nao))
    for k, (i, j) in enumerate(zip(*np.triu_indices(3))):
        d2AO_dr2[i,j] = d2AO_dr2[j,i] = ao[4+k]

    DM_AO = _dot_ao_dm(mol, AO, dm, None, shls_slice, ao_loc)
    rho = _contract_rho(AO, DM_AO)
    drho_dr = np.zeros((3, ngrids))
    d2rho_dr2 = np.zeros((3, 3, ngrids))

    for i in range(3):
        drho_dr[i] = _contract_rho(dAO_dr[i], DM_AO)
        DM_dAO_dr = _dot_ao_dm(mol, dAO_dr[i], dm, None, shls_slice, ao_loc)
        for j in range(i, 3):
            d2rho_dr2[i,j] = d2rho_dr2[j,i] = _contract_rho(dAO_dr[i], DM_dAO_dr)
    d2rho_dr2 += np.einsum('...ip,ip->...i', d2AO_dr2, DM_AO)
    d2rho_dr2 *= 2.0
    drho_dr   *= 2.0

    return rho, drho_dr, d2rho_dr2


def compute_dori(mol, dm, coords):
    r""" Inner function to compute DORI.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (2d array): density matrix
        grid_level (int): Controls the number of radial and angular points.

        mol : an instance of :class:`Mole`
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian)
        coords : 2D array of (3,ngrids)
            Grid coordinates

    Returns:
        1D array of (ngrids)
            Computed DORI

    Reference:
        J. Chem. Theory Comput. 2014, 10, 9, 3745–3756 (10.1021/ct500490b)

    Definitions:
        $$ \mathrm{DORI}(\vec r) \equiv \gamma(\vec r) = \frac{\theta(\vec r)}{1+\theta(\vec r)} $$
        $$ \theta = \frac{|\nabla (k^2)|^2}{|\vec k|^6} $$
        $$ \vec k(\vec r) = \frac{\nabla \rho(\vec r)}{\rho(\vec r)} $$

    Maths:
        $$
        \vec\nabla \left(\left|\frac{\vec\nabla \rho}{\rho}\right|^2\right)
        = \frac{2\left(\rho\cdot\vec\nabla\vec\nabla^\dagger\rho
        - \vec\nabla\rho \vec\nabla^\dagger\rho)\right)\vec\nabla\rho}{\rho^3}
        \equiv \vec\nabla \left(|\vec k|^2\right)
        = 2\left(\frac{\vec\nabla\vec\nabla^\dagger\rho}{\rho}-\vec k \vec k^\dagger\right)\vec k
        $$
    """

    ao_value = eval_ao(mol, coords, deriv=2)
    rho, drho_dr, d2rho_dr2 = eval_rho(mol, ao_value, dm)

    k = drho_dr / rho
    k2 = np.einsum('xi,xi->i', k, k)
    H = d2rho_dr2 / rho - np.einsum('xi,yi->xyi', k, k)
    dk2_dr = 2.0 * np.einsum('xyi,yi->xi', H, k)

    dk2_dr_square = np.einsum('xi,xi->i', dk2_dr, dk2_dr)
    theta = dk2_dr_square / k2**3
    gamma = theta / (1.0 + theta)
    # gamma = dk2_dr_square / (dk2_dr_square + k2**3)

    return gamma


def dori(mol, dm, grid_level=1):
    """Compute DORI

    Args:
        mol : an instance of :class:`Mole`
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian)

    Returns:
        1D array of (ngrids)
            Computed DORI

    """

    grid = make_grid_for_rho(mol, grid_level=grid_level)
    dori = compute_dori(mol, dm, grid.coords)

    return dori
