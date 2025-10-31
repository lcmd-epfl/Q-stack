import numpy as np

def first(mol, rho):
    """ Computes the transition dipole moments.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        rho (numpy ndarray): Density Matrix (trnasition if given ) or fitting coefficnts for the same matrix.

    Returns:
        A numpy ndarray with the transition dipole moments. If rho is a 1D matrix, returns the Decomposed/predicted transition dipole moments; if rho is a 2D matrix, returns the ab initio transition dipole moments.
    """
    if rho.ndim==1:
        return r_c(mol, rho) #coefficient
    elif rho.ndim==2:
        return r_dm(mol, rho) #matrix
    else:
        raise RuntimeError('Dimension mismatch')


def r_dm(mol, dm):
    """Computes the electric dipole moment from a density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): 2D density matrix in AO basis.

    Returns:
        numpy ndarray: Electric dipole moment vector (3 components).
    """
    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm)
    return el_dip

def r_c(mol, rho):
    """Computes the electric dipole moment from fitting coefficients.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        rho (numpy ndarray): 1D array of density-fitting coefficients.

    Returns:
        numpy ndarray: Electric dipole moment vector (3 components).

    Note:
        Currently only supports contracted basis sets.
    """
    r  = np.zeros(3)
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        coord = mol.atom_coords()[iat]
        for gto in mol._basis[q]:
            l, [a, c] = gto
            if(l==0):
                I0 = c * (2.0*np.pi/a)**0.75
                r   += I0 * rho[i] * coord
                i+=1
            elif(l==1):
                I1 = c * (2.0*np.pi)**0.75 / (a**1.25)
                r   += I1 * rho[i:i+3]
                i+=3
            else:
                i+=2*l+1
    return r

def r2_c(rho, mol):
    """Compute the zeroth ( :math:`<1>` ), first ( :math:`<r>` ), and second ( :math:`<r^{2}>`) moments of electron density distribution.

    .. math::

        <1> = \\int \\rho d r
        \\quad
        ;
        \\quad
        <r> = \\int \\hat{r} \\rho d r
        \\quad
        ;
        \\quad
        <r^{2}> = \\int \\hat{r}^{2} \\rho d r

    Args:
        rho (numpy ndarray): 1D array of density-fitting coefficients.
        mol (pyscf Mole): pyscf Mole object.

    Returns:
        tuple: Three values (N, r, r2) representing:
            - N (float): Zeroth moment (integrated density).
            - r (numpy ndarray): First moment (3-component dipole vector).
            - r2 (float): Second moment (mean square radius).

    Note:
        Currently only supports contracted basis sets.
    """

    N  = 0.0           # <1> zeroth
    r  = np.zeros(3)   # <r> first
    r2 = 0.0           # <r^2> second moments electron density distribution
    i=0
    for iat in range(mol.natm):
        q = mol._atom[iat][0]
        coord = mol.atom_coords()[iat]
        for gto in mol._basis[q]:
            l, [a, c] = gto
            if(l==0):
                I0 = c * (2.0*np.pi/a)**0.75
                I2 = c * 3.0 * (np.pi**0.75) / (a**1.75 * 2.0**0.25)
                N   += I0 * rho[i]
                r   += I0 * rho[i] * coord
                r2  += I0 * rho[i] * (coord@coord)
                r2  += I2 * rho[i]
                i+=1
            elif(l==1):
                I1 = c * (2.0*np.pi)**0.75 / (a**1.25)
                temp = I1 * rho[i:i+3]
                r   += temp
                r2  += 2.0*(temp@coord)
                i+=3
            else:
                i+=2*l+1
    return N, r, r2
