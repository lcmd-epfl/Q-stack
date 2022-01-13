from pyscf import dft

def get_converged_dm_RKS(mol, xc):
    """Performs SCF and returns density matrix, given pyscf mol object and an XC density functional.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        xc (str): XC functional for computation.

    Returns:
        numpy ndarray: density matrix in AO-basis.
    
    """

    # Prepare and run an RKS computation object
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 1
    mf.run()

    # print("Convergence: ",mf.converged)
    # print("Energy: ",mf.e_tot)

    # Make the one-particle density matrix in ao-basis
    dm = mf.make_rdm1()

    return dm
