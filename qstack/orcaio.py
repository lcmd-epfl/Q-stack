import warnings
import struct
import numpy as np
from qstack.mathutils.matrix import from_tril
from qstack.tools import reorder_ao


def read_density(mol, basename, directory='./', version=500, openshell=False, reorder_dest='pyscf'):
    """Reads densities from an ORCA output.

    Args:
        mol (pyscf Mole): pyscf Mole object.
        basename (str): Job name (without extension).
        version (int): ORCA version.
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

    if reorder_dest is not None:
        dm = np.array([reorder_ao(mol, i, src='orca', dest=reorder_dest) for i in dm])

    dm = np.squeeze(dm)
    return dm


def _parse_gbw(fname):
    """ Many thanks to
    https://pysisyphus.readthedocs.io/en/latest/_modules/pysisyphus/calculators/ORCA.html
    """

    def read_array(f, n, dtype):
        return np.frombuffer(f.read(dtype().itemsize * n), dtype=dtype)

    def read_mos():
        f.seek(24)
        offset = struct.unpack("<q", f.read(np.int64().itemsize))[0]    # int64: pointer to orbitals
        f.seek(offset)
        sets   = struct.unpack("<i", f.read(np.int32().itemsize))[0]    # int32: number of MO sets
        nao    = struct.unpack("<i", f.read(np.int32().itemsize))[0]    # int32: number of orbitals
        assert sets in (1,2)

        coefficients_ab = []
        energies_ab = []
        occupations_ab = []

        for i in range(sets):
            coefficients = read_array(f, nao*nao, np.float64).reshape(-1, nao)
            occupations  = read_array(f, nao,     np.float64)
            energies     = read_array(f, nao,     np.float64)
            irreps       = read_array(f, nao,     np.int32)
            cores        = read_array(f, nao,     np.int32)
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
            for iao in range(nao_at):
                l, _, ngto, _ = struct.unpack("<4i", f.read(4 * np.int32().itemsize))
                a = read_array(f, MAX_PRIMITIVES, np.float64)   # exponents
                c = read_array(f, MAX_PRIMITIVES, np.float64)   # coefficients
                ls[at].append(l)
        return ls

    with open(fname, "rb") as f:
        coefficients_ab, energies_ab, occupations_ab = read_mos()
        ls = read_basis()
        return coefficients_ab, energies_ab, occupations_ab, ls


def reorder_coeff_inplace(c_full, mol, reorder_dest='pyscf', ls_from_orca=None):
    def _reorder_coeff(c):
        # In ORCA, at least def2-SVP and def2-TZVP for 3d metals
        # are not sorted wrt angular momenta
        # TODO add the fix to read_density()   #if gto.basis._format_basis_name(mol.basis)=='def2tzvp':
        if ls_from_orca is not None:
            indices_full = np.arange(mol.nao)
            for iat, ls in ls_from_orca.items():
                indices = []
                i = 0
                for il, l in enumerate(ls):
                    indices.append((l, il, i + np.arange(2*l+1)))
                    i += 2*l+1
                indices = sorted(indices, key=lambda x: (x[0], x[1]))
                indices = np.array([j for i in indices for j in i[2]])
                ao_limits = mol.offset_ao_by_atom()[iat][2:]
                atom_slice = np.s_[ao_limits[0]:ao_limits[1]]
                indices_full[atom_slice] = indices[:] + ao_limits[0]
            for i in range(len(c)):
                c[:,i] = c[indices_full,i]


        for i in range(len(c)):
            c[:,i] = reorder_ao(mol, c[:,i], src='orca', dest=reorder_dest)
    [_reorder_coeff(c_full[i]) for i in range(c_full.shape[0])]


def read_gbw(mol, fname, reorder_dest='pyscf', sort_l=True):
    c, e, occ, ls = _parse_gbw(fname)

    ls = {iat: lsiat for iat, lsiat in ls.items() if np.any(lsiat!=sorted(lsiat))}
    if ls and not sort_l:
        msg = f'{fname}: basis set is not sorted wrt angular momenta for atoms # {list(ls.keys())} and is kept as is'
        warnings.warn(msg)

    if reorder_dest is not None:
        reorder_coeff_inplace(c, mol, reorder_dest, ls if (ls and sort_l) else None)
    return c, e, occ

