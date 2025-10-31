qstack.qml.slatm
================

Functions
---------

get\_mbtypes (qs, qml=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

pad\_zeros (slatm)
~~~~~~~~~~~~~~~~~~

(No docstring.)

get\_two\_body (i, mbtype, q, dist, r0=defaults.r0, rcut=defaults.rcut, sigma=defaults.sigma2, dgrid=defaults.dgrid2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

get\_three\_body (j, mbtype, q, r, dist, rcut=defaults.rcut, theta0=defaults.theta0, sigma=defaults.sigma3, dgrid=defaults.dgrid3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

get\_slatm (q, r, mbtypes, qml\_compatible=True, stack\_all=True, global\_repr=False, r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2, theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(No docstring.)

get\_slatm\_for\_dataset (molecules, progress=False, global\_repr=False, qml\_mbtypes=True, qml\_compatible=True, stack\_all=True, r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2, theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the (a)SLATM representation for a set of molecules.

    Reference:
        B. Huang, O. A. von Lilienfeld,
        "Quantum machine learning using atom-in-molecule-based fragments selected on the fly",
        Nat. Chem. 12, 945–951 (2020), doi:10.1038/s41557-020-0527-z.

    Args:
        molecules (Union(List[ase.Atoms], List[str]): pre-loaded ASE molecules or paths to the xyz files.
                  Alternatively, a list of any objects providing fields .numbers and .positions (Å)
        global_repr (bool): return molecular SLATM if True, return atomic SLATM (aSLATM) if False
        qml_mbtypes (bool): if True, mbtypes order should be identical as from QML (https://www.qmlcode.org/).
                            if False, the elements are sorted thus mbtype order can differ from QML in some cases
        qml_compatible (bool): if False, the local representation (global_repr=False) is condensed
        stack_all (bool): if stack the representations into one big ndarray

        rcut (float): radial cutoff (Å) for the 2- and 3-body terms
        r0 (float): grid range parameter (Å) [r0, rcut] for the 2-body term
        sigma2 (float): gaussian width for the 2-body term (Å)
        dgrid2 (float): grid spacing for the 2-body term (Å)
        theta0 (float): grid range parameter (°) [theta0, 180-theta0] for the 3-body term
        sigma3 (float): gaussian width for the 3-body term (°)
        dgrid3 (float): grid spacing for the 3-body term (°)

        progress (bool): if print progress bar

    Returns:
        ndrarray or List[List[ndarray]] containing the SLATM representation for each molecule.

get\_slatm\_rxn (reactions, progress=False, qml\_mbtypes=True, r0=defaults.r0, rcut=defaults.rcut, sigma2=defaults.sigma2, dgrid2=defaults.dgrid2, theta0=defaults.theta0, sigma3=defaults.sigma3, dgrid3=defaults.dgrid3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Computes the SLATM_d representation for a list of reactions.

    Reference:
        P. van Gerwen, A. Fabrizio, M. D. Wodrich, C. Corminboeuf,
        "Physics-based representations for machine learning properties of chemical reactions",
        Mach. Learn.: Sci. Technol. 3, 045005 (2022), doi:10.1088/2632-2153/ac8f1a.

    Args:
        reactions (List[rxn]): a list of rxn objects containing reaction information.
            rxn.reactants (List[ase.Atoms]) is a list of reactants (ASE molecules),
            rxn.products (List[ase.Atoms]) is a list of products.
        qml_mbtypes (bool): if True, mbtypes order should be identical as from QML (https://www.qmlcode.org/).
                            if False, the elements are sorted thus mbtype order can differ from QML in some cases

        rcut (float): radial cutoff (Å) for the 2- and 3-body terms
        r0 (float): grid range parameter (Å) [r0, rcut] for the 2-body term
        sigma2 (float): gaussian width for the 2-body term (Å)
        dgrid2 (float): grid spacing for the 2-body term (Å)
        theta0 (float): grid range parameter (°) [theta0, 180-theta0] for the 3-body term
        sigma3 (float): gaussian width for the 3-body term (°)
        dgrid3 (float): grid spacing for the 3-body term (°)

        progress (bool): if print progress bar

    Returns:
        ndrarray containing the SLATM_d representation for each reaction

.. note::
   Generated statically from source by gen_rst.py; no imports performed.
