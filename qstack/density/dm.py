from pyscf import dft

def get_converged_dm(mol, xc):
    """Performs restricted SCF and returns density matrix, given pyscf mol object and an XC density functional.

    Args:
        mol (pyscf Mole): pyscf Mole object used for the computation of the density matrix.
        xc (str): XC functional for computation.

    Returns:
        numpy ndarray: density matrix in AO-basis.
    
    """

    if mol.multiplicity == 1:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc
    print("Starting Kohn-Sham computation at "+str(mf.xc)+"/"+str(mol.basis)+" level.")
    mf.verbose = 1
    mf.kernel()

    print("Convergence: ",mf.converged)
    print("Energy: ",mf.e_tot)

    # Make the one-particle density matrix in ao-basis
    dm = mf.make_rdm1()

    return dm