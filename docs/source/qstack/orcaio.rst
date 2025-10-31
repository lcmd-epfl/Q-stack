qstack.orcaio
=============

Functions
---------

read\_input (fname, basis, ecp=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Read the structure from an Orca input (XYZ coordinates in simple format only)

    Note: we do not read basis set info from the file.
    TODO: read also %coords block?

    Args:
        fname (str) : path to file
        basis (str/dict) : basis name, path to file, or dict in the pyscf format
    Kwargs:
        ecp (str) : ECP to use

    Returns:
        pyscf Mole object.

read\_density (mol, basename, directory='./', version=500, openshell=False, reorder\_dest='pyscf')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Read densities from an ORCA output.

    Tested on Orca versions 4.0, 4.2, and 5.0.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        basename (str): Job name (without extension).
    Kwargs:
        directory (str) : path to the directory with the density files.
        version (int): ORCA version (400 for 4.0, 421 for 4.2, 500 for 5.0).
        openshell (bool): If read spin density in addition to the electron density.
        reorder_dest (str): Which AO ordering convention to use.

    Returns:
        A numpy 2darray containing the density matrix (openshell=False)
        or a numpy 3darray containing the density and spin density matrices (openshell=True).

\_parse\_gbw (fname)
~~~~~~~~~~~~~~~~~~~~

::

    Parse ORCA .gbw files.

    Many thanks to
    https://pysisyphus.readthedocs.io/en/latest/_modules/pysisyphus/calculators/ORCA.html

    Args:
        fname (str): path to the gbw file.

    Returns:
        numpy 3darray of (s,nao,nao) containing the density matrix
        numpy 2darray of (s,nao) containing the MO energies
        numpy 2darray of (s,nao) containing the MO occupation numbers
        dict of {int : [int]} with a list of basis functions angular momenta
                       for each atom (not for element!)

\_get\_indices (mol, ls\_from\_orca)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Get coefficient needed to reorder the AO read from Orca.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        ls_from_orca : dict of {int : [int]} with a list of basis functions
                       angular momenta for those atoms (not elements!)
                       whose basis functions are *not* sorted wrt to angular momenta.
                       The lists represent the Orca order.

    Returns:
        numpy int 1darray of (nao,) containing the indices to be used as
                c_reordered = c_orca[indices]

reorder\_coeff\_inplace (c\_full, mol, reorder\_dest='pyscf', ls\_from\_orca=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Reorder coefficient read from ORCA .gbw

    Args:
        c_full : numpy 3darray of (s,nao,nao) containing the MO coefficients
                 to reorder
        mol (pyscf Mole): pyscf Mole object.
    Kwargs:
        reorder_dest (str): Which AO ordering convention to use.
        ls_from_orca : dict of {int : [int]} with a list of basis functions
                       angular momenta for those atoms (not elements!)
                       whose basis functions are *not* sorted wrt to angular momenta.
                       The lists represent the Orca order.

read\_gbw (mol, fname, reorder\_dest='pyscf', sort\_l=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Read orbitals from an ORCA output.

    Tested on Orca versions 4.2 and 5.0.
    Limited for Orca version 4.0 (cannot read the basis set).

    Args:
        mol (pyscf Mole): pyscf Mole object.
        fname (str): path to the gbw file.
    Kwargs:
        reorder_dest (str): Which AO ordering convention to use.
        sort_l (bool): if sort the basis functions wrt angular momenta.
                       e.g. PySCF requires them sorted.

    Returns:
        numpy 3darray of (s,nao,nao) containing the MO coefficients
        numpy 2darray of (s,nao) containing the MO energies
        numpy 2darray of (s,nao) containing the MO occupation numbers
           s is 1 for closed-shell and 2 for open-shell computation.
           nao is number of atomic/molecular orbitals.

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
