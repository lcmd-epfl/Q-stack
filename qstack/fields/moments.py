import numpy as np
from qstack.compound import basis_flatten
from qstack.mathutils.array import safe_divide, scatter


def first(mol, rho):
    r"""Wrapper to compute the first moment of a molecular density needed for dipole moments.

    $$\int r \rho(r) dr$$

    Args:
        mol (pyscf Mole): pyscf Mole object.
        rho (numpy ndarray): 2D (mol.nao×mol.nao) density matrix or 1D (mol.nao) fitting coefficients.

    Returns:
        numpy ndarray: Electronic dipole moment vector (3 components).
    """
    if rho.ndim==1:
        return r2_c(mol, rho, moments=(1,))[0]
    elif rho.ndim==2:
        return r_dm(mol, rho)
    else:
        raise RuntimeError(f'Dimension mismatch {rho.shape}')


def r_dm(mol, dm):
    """Computes the first moment of a density matrix.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        dm (numpy ndarray): 2D density matrix in AO basis.

    Returns:
        numpy ndarray: Electronic dipole moment vector (3 components).
    """
    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm)
    return el_dip


def r2_c(mol, rho, moments=(0,1,2), per_atom=False):
    """Compute the zeroth ( :math:`<1>` ), first ( :math:`<r>` ), and second ( :math:`<r^{2}>`) moments of a fitted density.

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
        mol (pyscf Mole): pyscf Mole object.
        rho (numpy ndarray): 1D array of density-fitting coefficients. Can be None to compute AO integrals instead.
        moments (tuple): Moments to compute (0, 1, and/or 2).

    Returns:
        tuple: If rho!=None, values representing the requested moments, possibly containing:
            - float: Zeroth moment (integrated density).
            - numpy ndarray: First moment (3-component dipole vector).
            - float: Second moment (mean square radius).
            If rho is None, arrays representing the requested moments in AO basis so that
            they can be contracted with the coefficients usin (returned array)@(rho).

            if rho is None and per_atom is True:
            0st moment: (mol.nao, mol.natm)
            1st moment: (3, mol.nao, mol.natm)
            2nd moment: (mol.nao, mol.natm)

            if rho is None and per_atom is False:
            0st moment: (mol.nao,)
            1st moment: (3, mol.nao)
            2nd moment: (mol.nao,)

            if rho is not None and per_atom is True:
            0st moment: (mol.natm,)
            1st moment: (3, mol.natm)
            2nd moment: (mol.natm,)


    """

    if max(moments)>2:
        raise RuntimeError('Only moments 0, 1, and 2 are supported.')
    ret = {}

    (iat, l, _), (a, c) = basis_flatten(mol)
    coords = mol.atom_coords()[iat]

    idx_l0 = np.where(l==0)[0]
    ta = safe_divide(2.0*np.pi, a[idx_l0])**0.75
    I0 = (c[idx_l0] * ta).sum(axis=1)
    if rho is None:
        if 0 in moments:
            moments_ao = np.zeros(mol.nao)
            moments_ao[idx_l0] = I0
            if per_atom:
                ret[0] = scatter(moments_ao, iat)
            else:
                ret[0] = moments_ao

    else:
        t0 = rho[idx_l0] * I0
        if 0 in moments:
            if per_atom:
                ret[0] = np.zeros(mol.natm)
                np.add.at(ret[0], iat[idx_l0], t0)
            else:
                ret[0] = t0.sum()

    if 1 in moments or 2 in moments:
        idx_l1 = np.where(l==1)[0]
        I1 = (c[idx_l1] * safe_divide((2.0*np.pi)**0.75, a[idx_l1]**1.25)).sum(axis=1)
        mask = np.tile([[1,0,0,0,1,0,0,0,1]], len(I1)//3).reshape(-1,3).T
        I1 = I1*mask
        if rho is not None:
            t1 = (rho[idx_l1]*I1).T

    if 1 in moments:
        if rho is None:
            moments_ao = np.zeros((3, mol.nao))
            moments_ao[:,idx_l0] = I0 * coords[idx_l0].T
            moments_ao[:,idx_l1] = I1
            moments_ao = moments_ao
            if per_atom:
                ret[1] = scatter(moments_ao, iat)
            else:
                ret[1] = moments_ao
        else:
            if per_atom:
                ret[1] = np.zeros((3, mol.natm))
                np.add.at(ret[1], iat[idx_l0], coords[idx_l0] * t0[:,None])
                np.add.at(ret[1], iat[idx_l1], t1)
            else:
                ret[1] = (t0 * coords[idx_l0].T).sum(axis=1) \
                       + t1.sum(axis=0)

    if 2 in moments:
        I2 = (c[idx_l0] * ta * safe_divide(1.5, a[idx_l0])).sum(axis=1)
        if rho is None:
            moments_ao = np.zeros(mol.nao)
            moments_ao[idx_l0] = I2 + (I0 * (coords[idx_l0]**2).sum(axis=1))
            moments_ao[idx_l1] = 2.0 * (I1.T * coords[idx_l1]).sum(axis=1)
            if per_atom:
                ret[2] = scatter(moments_ao, iat)
            else:
                ret[2] = moments_ao

        else:
            if per_atom:
                ret[2] = np.zeros(mol.natm)
                np.add.at(ret[2], iat[idx_l0], t0 * (coords[idx_l0]**2).sum(axis=1))
                np.add.at(ret[2], iat[idx_l0], rho[idx_l0] * I2)
                np.add.at(ret[2], iat[idx_l1], 2.0 * (t1 * coords[idx_l1]).sum(axis=1))
            else:
                ret[2] = t0 @ (coords[idx_l0]**2).sum(axis=1) \
                       + rho[idx_l0] @ I2 \
                       + 2.0 * (t1 * coords[idx_l1]).sum()

    return tuple(ret[i] for i in moments)
