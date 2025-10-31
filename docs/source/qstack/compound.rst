qstack.compound
===============

::

    Module containing all the operations to load, transform, and save molecular objects.

Functions
---------

xyz\_comment\_line\_parser (line)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    reads the 'comment' line of a XYZ file, and tries to infer its meaning

xyz\_to\_mol (inp, basis='def2-svp', charge=None, spin=None, ignore=False, unit=None, ecp=None, parse\_comment=False, read\_string=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Reads a molecular file in xyz format and returns a pyscf Mole object.

    Args:
        inp (str): path of the xyz file to read / xyz fine contents if read_string==True
        basis (str or dict): Basis set.
        charge (int): Provide/override charge of the molecule.
        spin (int): Provide/override spin of the molecule (alpha electrons - beta electrons).
        ignore (bool): If assume molecule closed-shell an assign charge either 0 or -1
        unit (str): Provide/override units (Ang or Bohr)
        ecp (str) : ECP to use

    Returns:
        A pyscf Mole object containing the molecule information.

mol\_to\_xyz (mol, fout, fmt='xyz')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Converts a pyscf Mole object into a molecular file in xyz format.

    Args:
        pyscf Mole: pyscf Mole object.
        fout (str): Name (including path) of the xyz file to write.

    Returns:
        A file in xyz format containing the charge, total spin and molecular coordinates.

make\_auxmol (mol, basis, copy\_ecp=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Builds an auxiliary Mole object given a basis set and a pyscf Mole object.

    Args:
        mol (pyscf Mole): Original pyscf Mole object.
        basis (str or dict): Basis set.

    Returns:
        An auxiliary pyscf Mole object.

rotate\_molecule (mol, a, b, g, rad=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Rotate a molecule: transform nuclear coordinates given a set of Euler angles.

    Args:
        mol (pyscf Mole): Original pyscf Mole object.
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool) : Wheter the angles are in radians or not.


    Returns:
        A pyscf Mole object with transformed coordinates.

fragments\_read (frag\_file)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Loads fragement definition from a frag file.

    Args:
        frag_file (str): Name (including path) of the frag file to read.

    Returns:
        A list of arrays containing the fragments.

fragment\_partitioning (fragments, prop\_atom\_inp, normalize=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the contribution of each fragment.

    Args:
        fragments (numpy ndarray): Fragment definition
        prop_atom_inp (list of arrays or array): Coefficients densities.
        normalize (bool): Normalized fragment partitioning. Defaults to True.

    Returns:
        A list of arrays or an array containing the contribution of each fragment.

make\_atom (q, basis)
~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

singleatom\_basis\_enumerator (basis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Enumerates the different tensors of atomic orbitals within a 1-atom basis set
    Each tensor is a $2l+2$-sized group of orbitals that share a radial function and $l$ value.
    For each tensor, return the values of $l$, $n$ (an arbitrary radial-function counter that starts at 0),
    as well as AO range

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
