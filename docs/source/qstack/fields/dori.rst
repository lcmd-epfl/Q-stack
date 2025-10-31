qstack.fields.dori
==================

Functions
---------

eval\_rho\_dm (mol, ao, dm, deriv=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Calculate the electron density and the density derivatives.

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

eval\_rho\_df (ao, c, deriv=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Calculate the electron density and the density derivatives
        for a fitted density.

    Args:
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

compute\_rho (mol, coords, dm=None, c=None, deriv=2, eps=0.0001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Wrapper to calculate the electron density and the density derivatives.

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

compute\_s2rho (rho, d2rho\_dr2, eps=0.0001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute the sign of 2nd eigenvalue of density Hessian × density

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

compute\_dori (rho, drho\_dr, d2rho\_dr2, eps=0.0001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Inner function to compute DORI analytically

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

compute\_dori\_num (mol, coords, dm=None, c=None, eps=0.0001, dx=0.0001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Inner function to compute DORI seminumerically
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

dori\_on\_grid (mol, coords, dm=None, c=None, eps=0.0001, alg='analytical', mem=1, dx=0.0001, progress=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Wrapper to compute DORI on a given grid

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

dori (mol, dm=None, c=None, eps=0.0001, alg='analytical', grid\_type='dft', grid\_level=1, nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX\_MARGIN, cubename=None, dx=0.0001, mem=1, progress=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Compute DORI

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

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
