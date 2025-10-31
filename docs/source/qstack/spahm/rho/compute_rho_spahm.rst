qstack.spahm.rho.compute\_rho\_spahm
====================================

Functions
---------

spahm\_a\_b (rep\_type, mols, dms, bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod, elements=None, only\_m0=False, zeros=False, printlevel=0, auxbasis=defaults.auxbasis, model=defaults.model, pairfile=None, dump\_and\_exit=False, same\_basis=False, only\_z=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes SPAHM(a,b) representations for a set of molecules.

    Args:
        - rep_type (str) : the representation type ('atom' or 'bond' centered)
        - mols (list): the list of molecules (pyscf.Mole objects)
        - dms (list of numpy.ndarray): list of guess density matrices for each molecule
        - bpath (str): path to the directory containing bond-optimized basis-functions (.bas)
        - cutoff (float): the cutoff distance (angstrom) between atoms to be considered as bond
        - omods (list of str): the selected mode for open-shell computations
        - elements (list of str): list of all elements present in the set of molecules
        - only_m0 (bool): use only basis functions with `m=0`
        - zeros (bool): add zeros features for non-existing bond pairs
        - printlevel (int): level of verbosity
        - pairfile (str): path to the pairfile (if None, atom pairs are detected automatically)
        - dump_and_exit (bool): to save pairfile for the set of molecules (without generating representaitons)
        - same_basis (bool): to use the same bond-optimized basis function for all atomic pairs (ZZ.bas == CC.bas for any Z)
        - only_z (list of str): restrict the atomic representations to atom types in this list

    Returns:
        A numpy.ndarray with the atomic spahm-b representations for each molecule (Nmods,Nmolecules,NatomMax,Nfeatures).
        with:   - Nmods: the alpha and beta components of the representation
                - Nmolecules: the number of molecules in the set
                - NatomMax: the maximum number of atoms in one molecule
                - Nfeatures: the number of features (for each omods)

get\_repr (rep\_type, mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=None, pairfile=None, dump\_and\_exit=False, same\_basis=True, bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod, elements=None, only\_m0=False, zeros=False, split=False, printlevel=0, auxbasis=defaults.auxbasis, model=defaults.model, with\_symbols=False, only\_z=None, merge=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes and reshapes an array of SPAHM(a,b) representations

    Args:
        - rep_type (str) : the representation type ('atom' or 'bond' centered)
        - mols (list): the list of molecules (pyscf.Mole objects)
        - xyzlist (list of str): list with the paths to the xyz files
        - guess (str): the guess Hamiltonian
        - xc (str): the exchange-correlation functionals
        - dms (list of numpy.ndarray): list of guess density matrices for each molecule
        - readdm (str): path to the .npy file containins density matrices
        - bpath (str): path to the directory containing bond-optimized basis-functions (.bas)
        - cutoff (float): the cutoff distance (angstrom) between atoms to be considered as bond
        - omods (list of str): the selected mode for open-shell computations
        - spin (list of int): list of spins for each molecule
        - elements (list of str): list of all elements present in the set of molecules
        - only_m0 (bool): use only basis functions with `m=0`
        - zeros (bool): add zeros features for non-existing bond pairs
        - printlevel (int): level of verbosity
        - pairfile (str): path to the pairfile (if None, atom pairs are detected automatically)
        - dump_and_exit (bool): to save pairfile for the set of molecules (without generating representaitons)
        - same_basis (bool): to use the same bond-optimized basis function for all atomic pairs (ZZ.bas == CC.bas for any Z)
        - only_z (list of str): restrict the atomic representations to atom types in this list
        - split (bool): to split the final array into molecules
        - with_symbols (bool): to associate atomic symbol to representations in final array
        - merge (bool): to concatenate alpha and beta representations to a single feature vector

    Returns:
        A numpy.ndarray with all representations with shape (Nmods,Nmolecules,Natoms,Nfeatures)
        with:
          - Nmods: the alpha and beta components of the representation
          - Nmolecules: the number of molecules in the set
          - Natoms: the number of atoms in one molecule
          - Nfeatures: the number of features (for each omod)
        reshaped according to:
            - if split==False: collapses Nmolecules and returns a single np.ndarray (Nmods,Natoms,Nfeatures) (where Natoms is the total number of atoms in the set of molecules)
            - if merge==True: collapses the Nmods axis into the Nfeatures axis
            - if with_symbols==True: returns (for each molecule (Natoms, 2) containging the atom symbols along 1st dim and one of the above arrays

main (args=None)
~~~~~~~~~~~~~~~~

(No docstring.)

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
