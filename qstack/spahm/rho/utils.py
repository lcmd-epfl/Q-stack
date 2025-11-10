"""Utility functions for SPAHM(a,b) computation and default settings.

Provides:
    defaults: Default parameters for SPAHM(a,b) computation.
    omod_fns_dict: Dictionary of density matrix modification functions for open-shell systems.
"""

import os
import warnings
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import qstack.spahm.compute_spahm as spahm
from qstack.spahm import guesses
from qstack import compound


defaults = SimpleNamespace(
        guess='LB',
        model='Lowdin-long-x',
        basis='minao',
        auxbasis='ccpvdzjkfit',
        omod=['alpha', 'beta'],
        elements=["H", "C", "N", "O", "S"],
        cutoff=5.0,
        xc='hf',
        bpath=os.path.dirname(__file__)+'/basis_opt',
        )


def get_chsp(fname, n):
    """Load charge and spin information from file.

    Reads a file containing charge/spin values, converting 'None' strings to None objects.

    Args:
        fname (str or None): Path to charge/spin file. If None, returns array of Nones.
        n (int): Expected number of entries in the file.

    Returns:
        numpy ndarray: Array of charge/spin values (int or None) of length n.

    Raises:
        RuntimeError: If file is not found or has wrong length.
    """
    def chsp_converter(chsp):
        if chsp == 'None':
            chsp = None
        else:
            chsp = int(chsp)
        return chsp
    if fname is None:
        return np.full(n, None, dtype=object)
    if os.path.isfile(fname):
        chsp = np.loadtxt(fname, dtype=object, converters=chsp_converter, encoding=None)
        if(len(chsp)!=n):
            raise RuntimeError(f'Wrong length of the file {fname}')
    else:
        raise RuntimeError(f"{fname} can not be found")
    return chsp


def load_mols(xyzlist, charge, spin, basis, printlevel=0, units='ANG', ecp=None, progress=False, srcdir=None):
    """Load molecules from XYZ files and creates pyscf Mole objects.

    Args:
        xyzlist (list): List of XYZ filenames.
        charge (list or None): List of molecular charges (or None for neutral).
        spin (list or None): List of spin multiplicities (or None for default).
        basis (str or dict): Basis set.
        printlevel (int): Verbosity level (0=silent). Defaults to 0.
        units (str): Coordinate units ('ANG' or 'BOHR'). Defaults to 'ANG'.
        ecp (str or dict, optional): Effective core potential. Defaults to None.
        progress (bool): If True, shows progress bar. Defaults to False.
        srcdir (str, optional): Source directory prepended to XYZ filenames. Defaults to None.

    Returns:
        list: List of pyscf Mole objects.
    """
    mols = []
    if spin is None:
        spin = [None] * len(xyzlist)
    if charge is None:
        charge = [None] * len(xyzlist)
    for xyzfile, ch, sp in zip(tqdm(xyzlist, disable=not progress), charge, spin, strict=True):
        if srcdir is not None:
            xyzfile = srcdir+xyzfile
        if printlevel>0:
            print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis,
                                        charge=0 if ch is None else ch,
                                        spin=0 if sp is None else sp,
                                        unit=units, ecp=ecp))
    if printlevel>0:
        print()
    return mols


def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=None, printlevel=0):
    """Compute or loads guess density matrices for a list of molecules.

    Args:
        mols (list): List of pyscf Mole objects.
        xyzlist (list): List of XYZ filenames (for naming/loading).
        guess (str or callable): Guess method name or function.
        xc (str): Exchange-correlation functional for guess. Defaults to defaults.xc.
        spin (list or None): List of spin multiplicities. Defaults to None.
        readdm (str, optional): Directory path to load pre-computed density matrices. Defaults to None.
        printlevel (int): Verbosity level. Defaults to 0.

    Returns:
        list: List of density matrices (2D or 3D numpy arrays).
    """
    dms = []
    guess = guesses.get_guess(guess)
    if spin is None:
        spin = [None]*len(xyzlist)
    for xyzfile, mol, sp in zip(xyzlist, mols, spin, strict=True):
        if printlevel>0:
            print(xyzfile, flush=True)
        if readdm is None:
            _e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm = guesses.get_dm(v, mol.nelec, mol.spin if sp is not None else None)
        else:
            dm = np.load(f'{readdm}/{os.path.basename(xyzfile)}.npy')
            if spin and dm.ndim==2:
                dm = np.array((dm/2,dm/2))
        dms.append(dm)
        if printlevel>0:
            print()
    return dms


def dm_open_mod(dm, omod):
    """Treats density matrix according to the open-shell mode..

    Args:
        dm (numpy ndarray): Density matrix (2D for closed-shell, 3D for open-shell).
        omod (str or None): Open-shell mode. Options in omod_fns_dict.

    Returns:
        numpy ndarray: Modified density matrix.

    Raises:
        NotImplementedError: If omod is not a valid modification type.
        RuntimeError: If dm is 2D but omod is None, or if dm is 3D but omod is not None.
    """
    if omod is None:
        if dm.ndim==3:
            raise RuntimeError('Density matrix is open-shell (3D) but omod is None')
        elif dm.ndim==2:
            return dm
    elif dm.ndim == 2:
        raise RuntimeError('Density matrix is closed-shell (2D) but omod is not None')
    if omod not in omod_fns_dict:
        raise NotImplementedError(f'unknown open-shell mode: must be in {list(omod_fns_dict.keys())}, None if the system is closed-shell')
    return omod_fns_dict[omod](dm)


def get_xyzlist(xyzlistfile):
    """Load list of paths to files.

    Args:
        xyzlistfile (str): Path to the file containing list of XYZ filenames.

    Returns:
        numpy ndarray: Array of XYZ filenames as strings.
    """
    return np.loadtxt(xyzlistfile, dtype=str, ndmin=1)


def check_data_struct(fin, local=False):
    """Check the structure of a representation file.

    Args:
        fin (str): Input file path.
        local (bool): If True, checks for local representations.

    Returns:
        tuple: (is_single (bool), is_labeled (bool))
            is_single: True if the file contains a single representation.
            is_labeled: True if the representations are labeled.
    """
    x = np.load(fin, allow_pickle=True)
    if type(x.flatten()[0]) is str or type(x.flatten()[0]) is np.str_:
        is_labeled = True
        if not local and x.shape[0] == 1:
            is_single = True
        elif  x.shape[1] == 2:
            is_single = True
        else:
            is_single = False
    else:
        is_labeled = False
        if not local and x.ndim == 1:
            is_single = True
        elif x.shape[1] != 2: ## could be problematic! (if it's a set of local representations and nfeatures = 2 !!
            is_single=True
        else:
            is_single = False
    return is_single, is_labeled


def load_reps(f_in, from_list=True, srcdir=None, with_labels=False,
              local=True, sum_local=False, printlevel=0, progress=False,
              file_format=None):
    """Load representations from disk.

    Args:
        f_in (str): Path to the input file.
        from_list (bool): If the input file is a text file containing a list of paths to the representations.
        srcdir (str) : The path prefix to be at the begining of each file in `f_in`. Defaults to current working directory.
        with_labels (bool): If return atom type labes along with the representations.
        local (bool): If the representations are local (per-atom) or global (per-molecule).
        sum_local (bool): Sums the local components into a global representation, only if local=True.
        printlevel (int): Verbosity level.
        progress (bool): If shows a progress bar.
        file_format (dict): Structure of the input data, with keys=('is_labeled;, 'is_single').
            Defaults to structure auto determination (for "experienced users" only).

    Returns:
        np.array with shape (N_representations, N_features), or a tuple containing a list of atomic labels and said np.array.

    Raises:
        RuntimeError: In case of shape mismatch.
    """
    if file_format is None:  # Do not use mutable data structures for argument defaults
        file_format = {'is_labeled':None, 'is_single':None}
    if srcdir is None:
        path2list = os.getcwd()
    else:
        path2list = srcdir
    if from_list:
        X_list = get_xyzlist(f_in)
        Xs = []     # Xs must be a list of representations (local or global) whose len() = # of representations
        for f_X in X_list:
            if None in file_format.values():
                is_single, is_labeled = check_data_struct(os.path.join(path2list,f_X), local=local)
            else:
                is_single, is_labeled = file_format['is_single'], file_format['is_labeled']
            if printlevel > 0 :
                print(is_single, is_labeled)
            Xs.append(np.load(os.path.join(path2list,f_X), allow_pickle=True))
    else:
        if None in file_format.values():
            is_single, is_labeled = check_data_struct(os.path.join(path2list,f_in), local=local)
        else:
            is_single, is_labeled = file_format['is_single'], file_format['is_labeled']
        # if the given file contains a single representation create a one-element list
        Xs = [np.load(os.path.join(path2list,f_in), allow_pickle=True)] if is_single else np.load(os.path.join(path2list,f_in), allow_pickle=True)
    if printlevel > 1:
        print(f"Loading {len(Xs)} representations (local = {local}, labeled = {is_labeled})")
    reps = []
    labels = []
    for x in tqdm(Xs, disable=not progress):
        if local:
            if is_labeled:
                if sum_local:
                    reps.append(x[:,1].sum(axis=0))
                else:
                    reps.extend(x[:,1])
                    labels.extend(x[:,0])
            else:
                if sum_local:
                    reps.append(x.sum(axis=0))
                else:
                    reps.extend(x)
        else:
           if is_labeled:
                reps.append(x[1])
                labels.extend(x[0])
           else:
                reps.append(x)
    try:
        reps = np.array(reps, dtype=float)
    except ValueError as err:
        shapes = {r.shape[0] for r in reps}
        print(f'{len(reps)=} {shapes=}')
        raise RuntimeError(f"Error while loading representations in {f_in}, check the parameters") from err
    if with_labels and (len(labels) < len(reps)):
        warnings.warn("All labels could not be recovered (verify your representation files).", RuntimeWarning, stacklevel=2)
    if with_labels:
        return reps, labels
    else:
        return reps


def regroup_symbols(file_list, print_level=0, trim_reps=False):
    """Regroups representations by atom type.

    Args:
        file_list (list): List of representation files.
        print_level (int): Verbosity level. Defaults to 0.
        trim_reps (bool): If True, trims zeros from representations. Defaults to False.

    Returns:
        dict: Dictionary with atom types as keys and lists of representations as values.
    """
    reps, atoms = load_reps(file_list, from_list=True, with_labels=True, local=True, printlevel=print_level)
    if print_level > 0:
        print(f"Extracting {len(atoms)} atoms from {file_list}:")
    atoms_set = {e:[] for e in set(atoms)}
    for e, v in zip(atoms, reps, strict=True):
        if trim_reps:
            v = np.trim_zeros(v)
        atoms_set[e].append(v)
    if print_level > 0:
        print([(k, len(v)) for k, v in atoms_set.items()])
    return atoms_set


omod_fns_dict = {
        'sum':   lambda dm: dm[0]+dm[1],
        'diff':  lambda dm: dm[0]-dm[1],
        'alpha': lambda dm: dm[0],
        'beta':  lambda dm: dm[1],
        }
