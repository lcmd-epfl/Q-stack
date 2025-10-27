import warnings
import struct
import numpy as np
import pyscf
from qstack.mathutils.matrix import from_tril
from qstack.tools import reorder_ao


def read_input(fname, basis, ecp=None):
    """Read the structure from an Orca input (XYZ coordinates in simple format only)

    Note: we do not read basis set info from the file.
    TODO: read also %coords block?

    Args:
        fname (str) : path to file
        basis (str/dict) : basis name, path to file, or dict in the pyscf format
    Kwargs:
        ecp (str) : ECP to use

    Returns:
        pyscf Mole object.
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


def read_density(mol, basename, directory='./', version=500, openshell=False, reorder_dest='pyscf'):
    """Read densities from an ORCA output.

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


    is_def2 = 'def2' in pyscf.gto.basis._format_basis_name(mol.basis)
    has_3d = np.any([21 <= pyscf.data.elements.charge(q) <= 30 for q in mol.elements])
    if is_def2 and has_3d:
        msg = f'\n{path}:\nBasis set is not sorted wrt angular momenta for 3d elements.\nBetter use a gbw file.'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if reorder_dest is not None:
            msg = f'\nDensity matrix reordering for ORCA to {reorder_dest} is compromised.'
            raise RuntimeError(msg)

    if reorder_dest is not None:
        dm = np.array([reorder_ao(mol, i, src='orca', dest=reorder_dest) for i in dm])

    dm = np.squeeze(dm)
    return dm


def _parse_gbw(fname):
    """ Parse ORCA .gbw files.

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
    """ Get coefficient needed to reorder the AO read from Orca.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        ls_from_orca : dict of {int : [int]} with a list of basis functions
                       angular momenta for those atoms (not elements!)
                       whose basis functions are *not* sorted wrt to angular momenta.
                       The lists represent the Orca order.

    Returns:
        numpy int 1darray of (nao,) containing the indices to be used as
                c_reordered = c_orca[indices]
    """
    if ls_from_orca is None:
        return None
    ao_limits = mol.offset_ao_by_atom()[:,2:]
    indices_full = np.arange(mol.nao)
    for iat, ls in ls_from_orca.items():
        indices = []
        i = 0
        for il, l in enumerate(ls):
            indices.append((l, il, i + np.arange(2*l+1)))
            i += 2*l+1
        indices = sorted(indices, key=lambda x: (x[0], x[1]))
        indices = np.array([j for i in indices for j in i[2]])
        atom_slice = np.s_[ao_limits[iat][0]:ao_limits[iat][1]]
        indices_full[atom_slice] = indices[:] + ao_limits[iat][0]
    if np.all(sorted(indices_full)!=np.arange(mol.nao)):
        raise RuntimeError("Cannot reorder AOs")
    return indices_full


def reorder_coeff_inplace(c_full, mol, reorder_dest='pyscf', ls_from_orca=None):
    """ Reorder coefficient read from ORCA .gbw

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
    """
    def _reorder_coeff(c):
        # In ORCA, at least def2-SVP and def2-TZVP for 3d metals
        # are not sorted wrt angular momenta
        idx = _get_indices(mol, ls_from_orca)
        for i in range(len(c)):
            if idx is not None:
                c[:,i] = c[idx,i]
            c[:,i] = reorder_ao(mol, c[:,i], src='orca', dest=reorder_dest)
    for i in range(c_full.shape[0]):
        _reorder_coeff(c_full[i])


def read_gbw(mol, fname, reorder_dest='pyscf', sort_l=True):
    """Read orbitals from an ORCA output.

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
    return c, e, occ

