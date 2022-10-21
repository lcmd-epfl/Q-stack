import numpy as np
import pyscf
from qstack import compound, fields

def first(mol, rho):
    if rho.ndim==1:
        return r_c(mol, rho)
    elif rho.ndim==2:
        return r_dm(mol, rho)
    else:
        raise RuntimeError('Dimension mismatch')


def r_dm(mol, dm):
    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', ao_dip, dm)
    return el_dip

def r_c(mol, rho):
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
    N  = 0.0           # <1>
    r  = np.zeros(3)   # <r>
    r2 = 0.0           # <r^2>
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
