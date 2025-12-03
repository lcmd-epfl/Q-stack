"""ORCA quantum chemistry package I/O utilities.

Read and parse ORCA output files, including orbitals and densities binary files.
"""

import warnings
import struct
import numpy as np
import pyscf
from qstack.mathutils.matrix import from_tril
from qstack.reorder import reorder_ao
from qstack.tools import Cursor


def read_input(fname, basis, ecp=None):
    """Read the structure from an Orca input (XYZ coordinates in simple format only).

    Note: We do not read basis set info from the file.
    TODO: Read also %coords block?

    Args:
        fname (str): Path to Orca input file.
        basis (str or dict): Basis name, path to file, or dict in the pyscf format.
        ecp (str): Effective core potential to use. Defaults to None.

    Returns:
        pyscf.gto.Mole: pyscf Mole object.
    """
    with open(fname) as f:
        lines = [x.strip() for x in f]

    command_line = '\n'.join(y[1:] for y in filter(lambda x: x.startswith('!'), lines)).lower().split()
    if 'bohrs' in command_line:
        unit = 'Bohr'
    else:
        unit = 'Angstrom'

    idx_xyz_0, idx_xyz_1 = [y[0] for y in filter(lambda x: x[1].startswith('*'), enumerate(lines))][:2]
    charge, mult = map(int, lines[idx_xyz_0][1:].split()[1:])
    molxyz = '\n'.join(lines[idx_xyz_0+1:idx_xyz_1])

    mol = pyscf.gto.Mole()
    mol.atom = molxyz
    mol.charge = charge
    mol.spin = mult-1
    mol.unit = unit
    mol.basis = basis
    mol.ecp = ecp
    mol.build()
    return mol


def read_density(mol, basename, directory='./', version=500, openshell=False, reorder_dest='pyscf',
                 sort_l=True, ls=None):
    """Read densities from an ORCA output.

    Tested on Orca versions 4.0, 4.2, and 5.0.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        basename (str): Job name (without extension).
        directory (str): Path to the directory with the density files. Defaults to './'.
        version (int): ORCA version (400 for 4.0, 421 for 4.2, 500 for 5.0). Defaults to 500.
        openshell (bool): Whether to read spin density in addition to electron density. Defaults to False.
        reorder_dest (str): Which AO ordering convention to use. Defaults to 'pyscf'.
        sort_l (bool): Whether to sort basis functions wrt angular momenta.
            Has to be True for reorder_dest != None. Defaults to True.
        ls (dict): Dictionary mapping atom index to list of basis function angular momenta
            in Orca order for atoms whose basis functions are NOT sorted wrt angular momenta. Defaults to None.
            Can be obtained from read_gbw(return_ls=True) or made manually.
            If None, automatic detection is attempted for selected basis sets.

    Returns:
        numpy.ndarray: 2D array containing density matrix (openshell=False) or
            3D array containing density and spin density matrices (openshell=True).

    Raises:
        RuntimeError: If density matrix reordering is compromised:
            - Both reorder_dest and sort_l=False are set.
            - ls is provided and sort_l=False.
            - Basis set name is unknown, sort_l=True, and ls is not provided.
        NotImplementedError: If a def2-family basis set is used for which the order is not hardcoded, and ls is not provided.
    """
    path = directory+'/'+basename
    if version < 500:
        if version==400:
            offset = 0
        if version==421:
            offset = 12
        path = [path+'.scfp', path+'.scfr']
    else:
        path = [path+'.densities']

    if openshell is True:
        nspin = 2
    else:
        nspin = 1
        path = path[:1]

    if version < 500:
        dm = np.array([from_tril(np.fromfile(f, offset=offset)) for f in path])
    else:
        dm = np.fromfile(path[0], offset=8, count=mol.nao*mol.nao*nspin).reshape((nspin,mol.nao,mol.nao))

    if reorder_dest is not None and sort_l is False:
        msg = f'{path}: cannot both reorder the orbitals to {reorder_dest} and keep the basis set unsorted wrt angular momenta.'
        raise RuntimeError(msg)

    if ls is not None and sort_l is False:
        msg = f'{path}: {sort_l=} is incompatible with ls!=None.'
        raise RuntimeError(msg)

    if isinstance(mol.basis, str):
        basis_name = pyscf.gto.basis._format_basis_name(mol.basis)
        is_def2 = 'def2' in basis_name
    else:
        msg = f'\n{path}:\nUnknown basis set. Orbital order can be compromised.\nBetter use a gbw file.'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        is_def2 = False
        if sort_l is True and ls is None:
            msg = f'\n{path}:\nCannot sort the AO wrt angular momenta.\nBetter use a gbw file.'
            raise RuntimeError(msg)

    iat_3d = np.nonzero(np.array([21 <= pyscf.data.elements.charge(q) <= 30 for q in mol.elements]))[0]

    if ls is None:
        msg = f'\n{path}:\nUsing automatic sorting of AO wrt angular momenta.\nBetter use a gbw file.'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if is_def2:
            if basis_name == 'def2svp':
                ls_3d = [0]*5 + [1]*2 + [2]*1 + [1, 2, 3]
            elif basis_name == 'def2tzvp':
                ls_3d = [0]*6 + [1]*3 + [2]*3 + [1, 2, 3]
            else:
                msg = f'\n{path}:\nCannot determine AO order for 3d elements with {basis_name} basis.\nBetter use a gbw file.'
                raise NotImplementedError(msg)
            ls = dict.fromkeys(iat_3d, ls_3d)
        else:
            ls = {}
    else:
        msg = f'\n{path}:\nUsing provided angular momenta list to sort the AO wrt angular momenta.\nBetter use a gbw file.'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if sort_l:
        idx = _get_indices(mol, ls)
        idx = np.ix_(idx,idx)
        dm[:,:,:] = dm[:,*idx]
    else:
        if is_def2 and len(iat_3d)>0:
            msg = f'\n{path}:\nBasis set is not sorted wrt angular momenta for 3d elements.\nBetter use a gbw file.'
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if reorder_dest is not None:
        dm = np.array([reorder_ao(mol, i, src='orca', dest=reorder_dest) for i in dm])

    dm = np.squeeze(dm)
    return dm


def _parse_gbw(fname):
    """Parse ORCA .gbw files.

    Many thanks to
    https://pysisyphus.readthedocs.io/en/latest/_modules/pysisyphus/calculators/ORCA.html

    Args:
        fname (str): Path to the gbw file.

    Returns:
        tuple: A tuple containing:
        - coefficients_ab (numpy.ndarray): 3D array of shape (s,nao,nao) with MO coefficients.
        - energies_ab (numpy.ndarray): 2D array of shape (s,nao) with MO energies.
        - occupations_ab (numpy.ndarray): 2D array of shape (s,nao) with MO occupation numbers.
        - ls (dict): Dictionary mapping atom index to list of basis function angular momenta.
        s=1 for closed-shell and 2 for open-shell computation,
            nao is the number of atomic/molecular orbitals.

    Raises:
        RuntimeError: If number of MO sets is not 1 or 2.
    """
    def read_array(f, n, dtype):
        return np.frombuffer(f.read(dtype().itemsize * n), dtype=dtype)

    def read_mos():
        f.seek(24)
        offset = struct.unpack("<q", f.read(np.int64().itemsize))[0]    # int64: pointer to orbitals
        f.seek(offset)
        sets   = struct.unpack("<i", f.read(np.int32().itemsize))[0]    # int32: number of MO sets
        nao    = struct.unpack("<i", f.read(np.int32().itemsize))[0]    # int32: number of orbitals
        if sets not in (1,2):
            raise RuntimeError(f"Cannot interpret number of MO sets = {sets}")

        coefficients_ab = []
        energies_ab = []
        occupations_ab = []

        for _i in range(sets):
            coefficients = read_array(f, nao*nao, np.float64).reshape(-1, nao)
            occupations  = read_array(f, nao,     np.float64)
            energies     = read_array(f, nao,     np.float64)
            _irreps      = read_array(f, nao,     np.int32)
            _cores       = read_array(f, nao,     np.int32)
            coefficients_ab.append(coefficients)
            energies_ab.append(energies)
            occupations_ab.append(occupations)

        coefficients_ab = np.array(coefficients_ab)
        energies_ab     = np.array(energies_ab)
        occupations_ab  = np.array(occupations_ab)
        return coefficients_ab, energies_ab, occupations_ab

    def read_basis(MAX_PRIMITIVES=37):
        f.seek(16)
        offset = struct.unpack("<q", f.read(np.int64().itemsize))[0]
        f.seek(offset)
        _, nat = struct.unpack("<2i", f.read(2 * np.int32().itemsize))
        ls = {}
        for at in range(nat):
            ls[at] = []
            _, nao_at = struct.unpack("<2i", f.read(2 * np.int32().itemsize))
            for _iao in range(nao_at):
                l, _, _ngto, _ = struct.unpack("<4i", f.read(4 * np.int32().itemsize))
                _a = read_array(f, MAX_PRIMITIVES, np.float64)   # exponents
                _c = read_array(f, MAX_PRIMITIVES, np.float64)   # coefficients
                ls[at].append(l)
        return ls

    with open(fname, "rb") as f:
        coefficients_ab, energies_ab, occupations_ab = read_mos()
        try:
            ls = read_basis()
        except struct.error:
            ls = {}
        return coefficients_ab, energies_ab, occupations_ab, ls


def _get_indices(mol, ls_from_orca):
    """Get coefficients needed to reorder the AO read from Orca.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        ls_from_orca (dict): Dictionary mapping atom index to list of basis function angular momenta
            for atoms whose basis functions are NOT sorted wrt angular momenta.
            The lists represent the Orca order.

    Returns:
        numpy.ndarray: 1D integer array of shape (nao,) containing reordering indices.
            Use as: c_reordered = c_orca[indices]

    Raises:
        RuntimeError: If AO reordering fails.
    """
    if ls_from_orca is None:
        return None
    ao_limits = mol.offset_ao_by_atom()[:,2:]
    indices_full = np.arange(mol.nao)
    for iat, ls in ls_from_orca.items():
        i = Cursor(action='ranger')
        indices = [(l, il, i(2*l+1)) for il, l in enumerate(ls)]
        indices = sorted(indices, key=lambda x: (x[0], x[1]))
        indices = np.array([j for i in indices for j in i[2]])
        atom_slice = np.s_[ao_limits[iat][0]:ao_limits[iat][1]]
        indices_full[atom_slice] = indices[:] + ao_limits[iat][0]
    if np.all(sorted(indices_full)!=np.arange(mol.nao)):
        raise RuntimeError("Cannot reorder AOs")
    return indices_full


def reorder_coeff_inplace(c_full, mol, reorder_dest='pyscf', ls_from_orca=None):
    """Reorder coefficients read from ORCA .gbw in-place.

    Args:
        c_full (numpy.ndarray): 3D array of shape (s,nao,nao) containing MO coefficients to reorder.
        mol (pyscf.gto.Mole): pyscf Mole object.
        reorder_dest (str): Which AO ordering convention to use. Defaults to 'pyscf'.
        ls_from_orca (dict): Dictionary mapping atom index to list of basis function angular momenta
            for atoms whose basis functions are NOT sorted wrt angular momenta. Defaults to None.
    """
    def _reorder_coeff(c):
        # In ORCA, at least def2-SVP and def2-TZVP for 3d metals
        # are not sorted wrt angular momenta
        idx_from_l = _get_indices(mol, ls_from_orca)
        idx, sign = reorder_ao(mol, None, src='orca', dest=reorder_dest)
        for i in range(len(c)):
            if idx_from_l is not None:
                c[:,i] = c[idx_from_l,i]
            c[:,i] = c[idx,i]*sign
    for i in range(c_full.shape[0]):
        _reorder_coeff(c_full[i])


def read_gbw(mol, fname, reorder_dest='pyscf', sort_l=True, return_ls=False):
    """Read orbitals from an ORCA output.

    Tested on Orca versions 4.2 and 5.0.
    Limited for Orca version 4.0 (cannot read the basis set).

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        fname (str): Path to the gbw file.
        reorder_dest (str): Which AO ordering convention to use. Defaults to 'pyscf'.
        sort_l (bool): Whether to sort basis functions wrt angular momenta.
            PySCF requires them sorted. Defaults to True.
        return_ls (bool): Whether to return the dictionary mapping atom index to list of basis function
            angular momenta for atoms whose basis functions are NOT sorted wrt angular momenta. Defaults to False.

    Returns:
        tuple: A tuple containing:
        - c (numpy.ndarray): 3D array of shape (s,nao,nao) with MO coefficients.
        - e (numpy.ndarray): 2D array of shape (s,nao) with MO energies.
        - occ (numpy.ndarray): 2D array of shape (s,nao) with MO occupation numbers.
        Where s is 1 for closed-shell and 2 for open-shell computation,
            and nao is the number of atomic/molecular orbitals.

    Raises:
        RuntimeError: If basis set information not found and sort_l=True.
    """
    c, e, occ, ls = _parse_gbw(fname)
    if not ls and sort_l:
        raise RuntimeError(f'{fname}: basis set information not found. Cannot check if the basis set is sorted wrt angular momenta. Put sort_l=False to ignore')

    ls = {iat: lsiat for iat, lsiat in ls.items() if np.any(lsiat!=sorted(lsiat))}
    if ls and not sort_l:
        msg = f'\n{fname}: basis set is not sorted wrt angular momenta for atoms # {list(ls.keys())} and is kept as is'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if reorder_dest is not None:
        reorder_coeff_inplace(c, mol, reorder_dest, ls if (ls and sort_l) else None)
    if return_ls:
        return c, e, occ, ls
    else:
        return c, e, occ
