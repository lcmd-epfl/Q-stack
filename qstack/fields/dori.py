import numpy as np
from pyscf.dft.numint import eval_ao, _dot_ao_dm, _contract_rho
from pyscf.tools.cubegen import Cube, RESOLUTION, BOX_MARGIN
from .dm import make_grid_for_rho
from tqdm import tqdm


def eval_rho_dm(mol, ao, dm, deriv=2):
    r'''Calculate the electron density and the density derivatives.

    Taken from pyscf/dft/numint.py and modified to return second derivative matrices.

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
        ao : 3D array of shape (*,ngrids,nao):
            ao[0] : atomic oribitals values on the grid
            ao[1:4] : atomic oribitals derivatives values (if deriv>=1)
            ao[4:10] : atomic oribitals second derivatives values (if deriv==2)
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian)
    Kwargs:
        deriv : int
            Compute with up to `deriv`-order derivatives

    Returns:
        1D array of size ngrids to store electron density
        2D array of (3,ngrids) to store density derivatives (if deriv>=1)
        3D array of (3,3,ngrids) to store 2nd derivatives (if deriv==2)
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


def eval_rho_df(mol, ao, c, deriv=2):
    r'''Calculate the electron density and the density derivatives
        for a fitted density.

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
        ao : 3D array of shape (*,ngrids,nao):
            ao[0] : atomic oribitals values on the grid
            ao[1:4] : atomic oribitals derivatives values (if deriv>=1)
            ao[4:10] : atomic oribitals second derivatives values (if deriv==2)
        c : 1D array of (nao,)
            density fitting coefficients
    Kwargs:
        deriv : int
            Compute with up to `deriv`-order derivatives

    Returns:
        1D array of size ngrids to store electron density
        2D array of (3,ngrids) to store density derivatives (if deriv>=1)
        3D array of (3,3,ngrids) to store 2nd derivatives (if deriv==2)
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
    r'''Wrapper to calculate the electron density and the density derivatives.

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
        coords : 2D array of (ngrids,3)
            Grid coordinates (in Bohr)
    Kwargs:
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian) (confilicts with c)
        c : 1D array of (nao)
            density fitting coefficients (confilicts with dm)
        deriv : int
            Compute with up to `deriv`-order derivatives
        eps : float
            Min. density to compute the derivatives for

    Returns:
        1D array of size ngrids to store electron density
        2D array of (3,ngrids) to store density derivatives (if deriv>=1)
        3D array of (3,3,ngrids) to store 2nd derivatives (if deriv==2)
    '''
    if (c is None)==(dm is None):
        raise RuntimeError('Use either density matrix (dm) or density fitting coefficients (c)')
    if dm is not None:
        eval_rho = lambda ao, deriv: eval_rho_dm(mol, ao.reshape(-1, ao.shape[-2], ao.shape[-1]), dm, deriv=deriv)
    if c is not None:
        eval_rho = lambda ao, deriv: eval_rho_df(mol, ao.reshape(-1, ao.shape[-2], ao.shape[-1]), c, deriv=deriv)

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
    """Compute the sign of 2nd eigenvalue of density Hessian × density

    Args:
        rho : 1D array of (ngrids)
            Electron density
        d2rho_dr2 : 3D array of (3,3,ngrids)
            Density 2nd derivatives
    Kwargs:
        eps : float
            density threshold
    Returns:
        1D array of (ngrids) --- electron density * sgn(second eigenvalue of d^2rho/dr^2)
                                 if density>=eps else 0
    """
    s2rho = np.zeros_like(rho)
    idx = np.where(rho>=eps)
    s2rho[idx] = np.copysign(rho[idx], [sorted(np.linalg.eigh(h)[0])[1] for h in d2rho_dr2.T[idx]])
    return s2rho


def compute_dori(rho, drho_dr, d2rho_dr2, eps=1e-4):
    r""" Inner function to compute DORI analytically

    Args:
        rho : 1D array of (ngrids)
            Electron density
        drho_dr : 2D array of (3,ngrids)
            Density derivatives
        d2rho_dr2 : 3D array of (3,3,ngrids)
            Density 2nd derivatives
    Kwargs:
        eps : float
            Density threshold (if |rho|<eps then dori=0)

    Returns:
        1D array of (ngrids): DORI

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
    r""" Inner function to compute DORI seminumerically
    See documentation to compute_dori().

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
        coords : 2D array of (ngrids,3)
            Grid coordinates (in Bohr)
    Kwargs:
        dm : 2D array of (nao,nao)
            Density matrix (assumed Hermitian) (confilicts with c)
        c : 1D array of (nao)
            density fitting coefficients (confilicts with dm)
        eps : float
            Density threshold (if |rho|<eps then dori=0)
        dx : float
            Step (in Bohr) to take the numerical derivatives

    Returns:
        1D array of (ngrids): DORI
        1D array of (ngrids): electron density
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
    """Wrapper to compute DORI on a given grid

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
        coords : 2D array of (ngrids,3)
            Grid coordinates (in Bohr)
    Kwargs:
        dm : 2D array of (nao,nao)
            Density matrix (confilicts with c)
        c : 1D array of (nao)
            Density fitting coefficients (confilicts with dm)
        eps : float
            density threshold for DORI
        alg : str
            [a]nalytical or [n]umerical computation
        dx : float
            Step (in Bohr) to take the numerical derivatives
        mem : float
            max. memory (GiB) that can be allocated to compute
            the AO and their derivatives
        progress : bool
            if print a progress bar

    Returns:
        1D array of (ngrids) --- computed DORI
        1D array of (ngrids) --- electron density
        1D array of (ngrids) --- electron density * sgn(second eigenvalue of d^2rho/dr^2)
                                 if density>=eps else 0 (only with alg='analytical').
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
    """Compute DORI

    Args:
        mol : an instance of :class:`pyscf.gto.Mole`
    Kwargs:
        dm : 2D array of (nao,nao)
            Density matrix (confilicts with c)
        c : 1D array of (nao)
            Density fitting coefficients (confilicts with dm)
        eps : float
            density threshold for DORI
        alg : str
            [a]nalytical or [n]umerical computation
        grid_type : str
            Type of grid, 'dft' for a DFT grid and 'cube' for a cubic grid.
        grid_level : int
            For a DFT grid, the grid level.
        nx, ny, nz : int
            For a cubic grid,
            the number of grid point divisions in x, y, z directions.
            Conflicts to keyword resolution.
        resolution: float
            For a cubic grid,
            the resolution of the mesh grid in the cube box.
            Conflicts to keywords nx, ny, nz.
        cubename : str
            For a cubic grid,
            name for the cube files to save the results to.
        mem : float
              max. memory (GiB) that can be allocated to compute
              the AO and their derivatives
        dx : float
            Step (in Bohr) to take the numerical derivatives
        progress : bool
            if print a progress bar

    Returns:
        Tuple of:
            1D array of (ngrids) --- computed DORI
            1D array of (ngrids) --- electron density
            1D array of (ngrids) --- electron density * sgn(second eigenvalue of d^2rho/dr^2)
                                     if density>=eps else 0 (only with alg='analytical').
            2D array of (ngrids,3) --- grid coordinates
            1D array of (ngrids) --- grid weights

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
