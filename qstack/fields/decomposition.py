import numpy as np

def get_integrals(mol, auxmol):
    """Computes overlap and 2-/3-centers ERI matrices.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        auxmol (pyscf Mole): pyscf Mole object holding molecular structure, composition and the auxiliary basis set.

    Returns:
        numpy ndarray: Overlap matrix, 2-centers and 3-centers ERI matrices.
        
    """

    # Get overlap integral in the auxiliary basis
    S = auxmol.intor('int1e_ovlp_sph')

    # Concatenate standard and auxiliary basis set into a pmol object
    pmol = mol + auxmol

    # Compute 2- and 3-centers ERI integrals using the concatenated mol object
    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

    return S, eri2c, eri3c

def get_coeff(dm, eri2c, eri3c):
    """Computes the density expansion coefficients.

    Args:
        dm (numpy ndarray): density matrix.
        eri2c (numpy ndarray): 2-centers ERI matrix.
        eri3c (numpy ndarray): 3-centers ERI matrix.

    Returns:
        numpy ndarray: Expansion coefficients of the density onto the auxiliary basis.
        
    """

    # Compute the projection of the density onto auxiliary basis using a Coulomb metric
    projection = np.einsum('ijp,ij->p', eri3c, dm)

    # Solve Jc = projection to get the coefficients
    c = np.linalg.solve(eri2c, projection)

    return c


def number_of_electrons_deco(auxmol, c):
    """Computes the number of electrons of a molecule given a set of expansion coefficients and a Mole object.

    Args:
        auxmol (pyscf Mole): pyscf mol object holding molecular structure, composition and the auxiliary basis set.
        c (numpy ndarray): expansion coefficients of the density onto the auxiliary basis.

    Returns:
        int: number of electrons.
    """

    # Initialize variables 
    nel = 0.0
    i = 0

    # Loop over each atom in the molecule
    for iat in range(auxmol.natm):
        j = 0
        q = auxmol._atom[iat][0]
        max_l = auxmol._basis[q][-1][0]
        numbs = [x[0] for x in auxmol._basis[q]]

        # Loop over all radial channels for l = 0
        for n in range(numbs.count(0)):
            a, w = auxmol._basis[q][j][1]
            nel += c[i] * w * pow (2.0*np.pi/a, 0.75) # norm = (2.0*a/np.pi)^3/4, integral = (pi/a)^3/2
            i += 1
            j += 1
        # if l !=0 it does not contribute to the number of electrons, so skip them.
        for l in range(1,max_l+1):
            n_l = numbs.count(l)
            i += n_l * (2*l+1)
            j += n_l

    return nel
