import numpy as np
from pyscf.dft.numint import eval_ao, _dot_ao_dm, _contract_rho
from pyscf.tools.cubegen import Cube, RESOLUTION, BOX_MARGIN
from .dm import make_grid_for_rho
from tqdm import tqdm


def eval_rho_dm(mol, ao, dm, deriv=2):
    r'''Calculate electron density and its derivatives from a density matrix.

    Modified from pyscf/dft/numint.py to return full second derivative matrices
    needed for DORI calculations.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        ao (numpy ndarray): 3D array of shape (nderiv, ngrids, nao) where:
            - ao[0]: atomic orbital values on the grid
            - ao[1:4]: first derivatives (if deriv>=1)
            - ao[4:10]: second derivatives in triangular form (if deriv==2)
        dm (numpy ndarray): 2D array (nao, nao) - Hermitian density matrix in AO basis.
        deriv (int): Maximum derivative order to compute (0, 1, or 2). Defaults to 2.

    Returns:
        tuple: Depending on deriv value:
            - deriv=0: rho (1D array of size ngrids)
            - deriv=1: (rho, drho_dr) where drho_dr is (3, ngrids)
            - deriv=2: (rho, drho_dr, d2rho_dr2) where d2rho_dr2 is (3, 3, ngrids)
    '''

    AO, dAO_dr, d2AO_dr2 = np.split(ao, [1,4])
    DM_AO = _dot_ao_dm(mol, AO[0], dm, None, None, None)
    rho = _contract_rho(AO[0], DM_AO)
    if deriv==0:
        return rho

    drho_dr = np.zeros((3, len(rho)))
    if deriv==2:
        d2rho_dr2 = np.zeros((3, 3, len(rho)))
        triu_idx = {(i,j): k for k, (i,j) in enumerate(zip(*np.triu_indices(3), strict=True))}

    for i in range(3):
        drho_dr[i] = 2.0*_contract_rho(dAO_dr[i], DM_AO)
        if deriv==2:
            DM_dAO_dr_i = 2 * _dot_ao_dm(mol, dAO_dr[i], dm, None, None, None)
            for j in range(i, 3):
                d2rho_dr2[i,j] = _contract_rho(dAO_dr[j], DM_dAO_dr_i) + 2.0*np.einsum('ip,ip->i', d2AO_dr2[triu_idx[(i,j)]], DM_AO)
                d2rho_dr2[j,i] = d2rho_dr2[i,j]

    if deriv==1:
        return rho, drho_dr
    return rho, drho_dr, d2rho_dr2


def eval_rho_df(ao, c, deriv=2):
    r'''Calculate electron density and its derivatives from density-fitting coefficients.

    Computes density and derivatives directly from fitted/decomposed density
    representation using expansion coefficients.

    Args:
        ao (numpy ndarray): 3D array of shape (nderiv, ngrids, nao) where:
            - ao[0]: atomic orbital values on the grid
            - ao[1:4]: first derivatives (if deriv>=1)
            - ao[4:10]: second derivatives in triangular form (if deriv==2)
        c (numpy ndarray): 1D array of density fitting/expansion coefficients.
        deriv (int): Maximum derivative order to compute (0, 1, or 2). Defaults to 2.

    Returns:
        tuple: Depending on deriv value:
            - deriv=0: rho (1D array of size ngrids)
            - deriv=1: (rho, drho_dr) where drho_dr is (3, ngrids)
            - deriv=2: (rho, drho_dr, d2rho_dr2) where d2rho_dr2 is (3, 3, ngrids)
    '''

    maxdim = 1 if deriv==0 else (4 if deriv==1 else 10)
    rho_all = np.tensordot(ao[:maxdim], c, 1)  # corresponds to np.einsum('xip,p->xi', ao[:maxdim], c)
    if deriv==0:
        return rho_all[0]
    if deriv==1:
        return rho_all[0], rho_all[1:4]
    if deriv==2:
        d2rho_dr2 = np.zeros((3, 3, rho_all.shape[-1]))
        for k, (i, j) in enumerate(zip(*np.triu_indices(3), strict=True)):
            d2rho_dr2[i,j] = d2rho_dr2[j,i] = rho_all[4+k]
        return rho_all[0], rho_all[1:4], d2rho_dr2


def compute_rho(mol, coords, dm=None, c=None, deriv=2, eps=1e-4):
    r'''Wrapper to calculate electron density and derivatives efficiently.

    Computes density and its spatial derivatives on a grid from either a density
    matrix or fitting coefficients, with optimizations for numerical stability.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        coords (numpy ndarray): 2D array (ngrids, 3) of grid coordinates in Bohr.
        dm (numpy ndarray, optional): 2D density matrix in AO basis. Conflicts with c.
        c (numpy ndarray, optional): 1D density fitting coefficients. Conflicts with dm.
        deriv (int): Maximum derivative order (0, 1, or 2). Defaults to 2.
        eps (float): Minimum density threshold below which derivatives are set to zero. Defaults to 1e-4.

    Returns:
        tuple: Depending on deriv value:
            - deriv=0: rho (1D array)
            - deriv=1: (rho, drho_dr) where drho_dr is (3, ngrids)
            - deriv=2: (rho, drho_dr, d2rho_dr2) where d2rho_dr2 is (3, 3, ngrids)

    Raises:
        RuntimeError: If both or neither of dm and c are provided.
    '''
    if (c is None)==(dm is None):
        raise RuntimeError('Use either density matrix (dm) or density fitting coefficients (c)')
    if dm is not None:
        eval_rho = lambda ao, deriv: eval_rho_dm(mol, ao.reshape(-1, ao.shape[-2], ao.shape[-1]), dm, deriv=deriv)
    if c is not None:
        eval_rho = lambda ao, deriv: eval_rho_df(ao.reshape(-1, ao.shape[-2], ao.shape[-1]), c, deriv=deriv)

    ao0 = eval_ao(mol, coords, deriv=0)
    rho = eval_rho(ao0, deriv=0)
    if deriv==0:
        return rho
    good_idx = np.where(rho>=eps)[0]
    drho_dr = np.zeros((3,len(coords)))
    if deriv==2:
        d2rho_dr2 = np.zeros((3,3,len(coords)))
    if len(good_idx)>0:
        ao = eval_ao(mol, coords[good_idx], deriv=deriv)
        ret = eval_rho(ao, deriv=deriv)
        drho_dr[:,good_idx] = ret[1]
        if deriv==2:
            d2rho_dr2[:,:,good_idx] = ret[2]
    if deriv==1:
        return rho, drho_dr
    return rho, drho_dr, d2rho_dr2


def compute_s2rho(rho, d2rho_dr2, eps=1e-4):
    """Computes signed density based on second eigenvalue of the density Hessian.

    Useful for distinguishing bonding vs. non-bonding regions. The sign of the
    second eigenvalue of the Hessian indicates local density topology.

    Args:
        rho (numpy ndarray): 1D array (ngrids,) of electron density values.
        d2rho_dr2 (numpy ndarray): 3D array (3, 3, ngrids) of density second derivatives (Hessian).
        eps (float): Density threshold below which values are set to zero. Defaults to 1e-4.

    Returns:
        numpy ndarray: 1D array (ngrids,) containing rho * sign(λ₂) where λ₂ is the
            second eigenvalue of the Hessian, or 0 where rho < eps.
    """
    s2rho = np.zeros_like(rho)
    idx = np.where(rho>=eps)
    s2rho[idx] = np.copysign(rho[idx], [sorted(np.linalg.eigh(h)[0])[1] for h in d2rho_dr2.T[idx]])
    return s2rho


def compute_dori(rho, drho_dr, d2rho_dr2, eps=1e-4):
    r"""Computes Density Overlap Regions Indicator (DORI) analytically.

    DORI is a density-based descriptor for identifying covalent bonding regions,
    with values close to 1 indicating strong electron sharing (covalent bonds).

    Args:
        rho (numpy ndarray): 1D array (ngrids,) of electron density.
        drho_dr (numpy ndarray): 2D array (3, ngrids) of density first derivatives.
        d2rho_dr2 (numpy ndarray): 3D array (3, 3, ngrids) of density second derivatives.
        eps (float): Density threshold below which DORI is set to zero. Defaults to 1e-4.

    Returns:
        numpy ndarray: 1D array (ngrids,) of DORI values ranging from 0 to 1.

    Reference:
        J. Chem. Theory Comput. 2014, 10, 9, 3745–3756 (10.1021/ct500490b)

    Note:
        DORI is defined as:
            DORI(r) = γ(r) = θ(r) / (1 + θ(r))
        where:
            θ = |∇(k²)|² / |k|⁶
            k(r) = ∇ρ(r) / ρ(r)
    """

    idx = np.where(abs(rho)>=eps)[0]
    k = drho_dr[...,idx] / rho[idx]
    k2 = np.einsum('xi,xi->i', k, k)
    H = d2rho_dr2[...,idx] / rho[idx] - np.einsum('xi,yi->xyi', k, k)
    dk2_dr = 2.0 * np.einsum('xyi,yi->xi', H, k)

    dk2_dr_square = np.einsum('xi,xi->i', dk2_dr, dk2_dr)
    theta = dk2_dr_square / k2**3
    gamma = theta / (1.0 + theta)
    # gamma = dk2_dr_square / (dk2_dr_square + k2**3)

    gamma_full = np.zeros_like(rho)
    gamma_full[idx] = gamma
    return gamma_full


def compute_dori_num(mol, coords, dm=None, c=None, eps=1e-4, dx=1e-4):
    r"""Computes DORI using numerical differentiation (semi-numerical approach).

    Alternative to analytical DORI calculation using finite differences for
    derivatives of k². Useful for validation or when analytical gradients
    are problematic.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        coords (numpy ndarray): 2D array (ngrids, 3) of grid coordinates in Bohr.
        dm (numpy ndarray, optional): 2D density matrix in AO basis. Conflicts with c.
        c (numpy ndarray, optional): 1D density fitting coefficients. Conflicts with dm.
        eps (float): Density threshold below which DORI is zero. Defaults to 1e-4.
        dx (float): Finite difference step size in Bohr. Defaults to 1e-4.

    Returns:
        tuple: (dori, rho) where both are 1D arrays (ngrids,) containing
            DORI values and electron density respectively.
    """

    def grad_num(func, grid, d, **kwargs):
        g = np.zeros_like(grid)
        for j in range(3):
            u = np.eye(1, 3, j)  # unit vector || jth dimension
            e1  = func(grid+d*u, **kwargs)
            e2  = func(grid-d*u, **kwargs)
            e11 = func(grid+2*d*u, **kwargs)
            e22 = func(grid-2*d*u, **kwargs)
            g[:,j] = (8.0*e1-8.0*e2 + e22-e11) / (12.0*d)
        return g

    def compute_k2(coords, mol=None, dm=None, c=None):
        rho, drho_dr = compute_rho(mol, coords, dm=dm, c=c, deriv=1)
        k = drho_dr / rho
        return np.einsum('xi,xi->i', k, k)

    rho = compute_rho(mol, coords, dm=dm, c=c, deriv=0)

    good_idx = np.where(rho>=eps)[0]
    dori = np.zeros_like(rho)
    if len(good_idx):
        k2 = compute_k2(coords[good_idx], mol=mol, dm=dm, c=c)
        dk2_dr = grad_num(compute_k2, coords[good_idx], d=dx, mol=mol, dm=dm, c=c)
        dk2_dr_square = np.einsum('ix,ix->i', dk2_dr, dk2_dr)
        theta = dk2_dr_square / k2**3
        dori[good_idx] = theta / (theta + 1.0)
    return dori, rho


def dori_on_grid(mol, coords, dm=None, c=None, eps=1e-4, alg='analytical', mem=1, dx=1e-4, progress=False):
    """Computes DORI on a user-specified grid with memory management.

    Main computational function for DORI evaluation. Handles large grids by
    chunking based on available memory.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        coords (numpy ndarray): 2D array (ngrids, 3) of grid coordinates in Bohr.
        dm (numpy ndarray, optional): 2D density matrix in AO basis. Conflicts with c.
        c (numpy ndarray, optional): 1D density fitting coefficients. Conflicts with dm.
        eps (float): Density threshold for DORI calculation. Defaults to 1e-4.
        alg (str): Algorithm choice: 'analytical' or 'numerical'. Defaults to 'analytical'.
        mem (float): Maximum memory in GiB for AO evaluation. Defaults to 1.
        dx (float): Step size in Bohr for numerical derivatives. Defaults to 1e-4.
        progress (bool): If True, displays progress bar. Defaults to False.

    Returns:
        tuple: (dori, rho, s2rho) where:
            - dori: 1D array (ngrids,) of DORI values
            - rho: 1D array (ngrids,) of electron density
            - s2rho: 1D array (ngrids,) of signed density (None if numerical)
    """

    max_size = int(mem * 2**30)  # mem * 1 GiB
    point_size = 10 * mol.nao * np.float64().itemsize  # memory needed for 1 grid point
    dgrid = max_size // point_size
    grid_chunks = range(0, len(coords), dgrid)
    if progress:
        grid_chunks = tqdm(grid_chunks)

    rho = np.zeros(len(coords))

    if 'analytical'.startswith(alg.lower()):
        drho_dr = np.zeros((3, len(coords)))
        d2rho_dr2 = np.zeros((3, 3, len(coords)))
        for i in grid_chunks:
            s = np.s_[i:i+dgrid]
            rho[s], drho_dr[:,s], d2rho_dr2[:,:,s] = compute_rho(mol, coords[s], dm=dm, c=c, eps=eps)
        dori = compute_dori(rho, drho_dr, d2rho_dr2, eps=eps)
        s2rho = compute_s2rho(rho, d2rho_dr2, eps=eps)
        return dori, rho, s2rho

    elif 'numerical'.startswith(alg.lower()):
        dori = np.zeros_like(rho)
        for i in grid_chunks:
            s = np.s_[i:i+dgrid]
            dori_i, rho_i = compute_dori_num(mol, coords[s], dm=dm, c=c, eps=eps, dx=dx)
            dori[s] = dori_i
            rho[s] = rho_i
        return dori, rho, None


def dori(mol, dm=None, c=None,
         eps=1e-4, alg='analytical',
         grid_type='dft',
         grid_level=1,
         nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN,
         cubename=None,
         dx=1e-4, mem=1, progress=False):
    """High-level interface to compute DORI with automatic grid generation and file output.

    Computes the Density Overlap Regions Indicator (DORI) for analyzing chemical
    bonding. Automatically generates appropriate grids and optionally saves results
    to cube files for visualization.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        dm (numpy ndarray, optional): 2D density matrix in AO basis. Conflicts with c.
        c (numpy ndarray, optional): 1D density fitting coefficients. Conflicts with dm.
        eps (float): Density threshold for DORI. Defaults to 1e-4.
        alg (str): Algorithm: 'analytical' or 'numerical'. Defaults to 'analytical'.
        grid_type (str): Grid type: 'dft' for DFT quadrature grid or 'cube' for uniform grid. Defaults to 'dft'.
        grid_level (int): For DFT grid, the grid level (higher = more points). Defaults to 1.
        nx (int): For cube grid, number of points in x direction. Defaults to 80.
        ny (int): For cube grid, number of points in y direction. Defaults to 80.
        nz (int): For cube grid, number of points in z direction. Defaults to 80.
        resolution (float): For cube grid, grid spacing in Bohr. Conflicts with nx/ny/nz.
        margin (float): For cube grid, extra space around molecule in Bohr. Defaults to BOX_MARGIN.
        cubename (str, optional): For cube grid, base filename for output cube files. If None, no files saved.
        dx (float): For numerical algorithm, finite difference step in Bohr. Defaults to 1e-4.
        mem (float): Maximum memory in GiB for AO evaluation. Defaults to 1.
        progress (bool): If True, displays progress bar. Defaults to False.

    Returns:
        tuple: (dori, rho, s2rho, coords, weights) containing:
            - dori (numpy ndarray): 1D array of DORI values
            - rho (numpy ndarray): 1D array of electron density
            - s2rho (numpy ndarray): 1D array of signed density (None if numerical)
            - coords (numpy ndarray): 2D array (ngrids, 3) of grid coordinates
            - weights (numpy ndarray): 1D array of grid weights

    Note:
        When cubename is provided with cube grid, creates three files:
        - <cubename>.dori.cube: DORI values
        - <cubename>.rho.cube: electron density
        - <cubename>.sgnL2rho.cube: signed density (analytical only)
    """

    if grid_type=='dft':
        grid = make_grid_for_rho(mol, grid_level=grid_level)
        weights = grid.weights
        coords  = grid.coords
    elif grid_type=='cube':
        grid = Cube(mol, nx, ny, nz, resolution, margin)
        weights = np.full(grid.get_ngrids(), np.prod(np.diag(grid.box)) * grid.get_volume_element())
        coords  = grid.get_coords()

    dori, rho, s2rho = dori_on_grid(mol, coords, dm=dm, c=c, eps=eps, alg=alg, mem=mem, dx=dx, progress=progress)

    if grid_type=='cube' and cubename:
        grid.write(dori.reshape(grid.nx, grid.ny, grid.nz), cubename+'.dori.cube', comment='DORI')
        grid.write(rho.reshape(grid.nx, grid.ny, grid.nz), cubename+'.rho.cube', comment='electron density rho')
        if s2rho is not None:
            grid.write(s2rho.reshape(grid.nx, grid.ny, grid.nz), cubename+'.sgnL2rho.cube', comment='sgn(lambda_2)*rho')

    return dori, rho, s2rho, coords, weights
