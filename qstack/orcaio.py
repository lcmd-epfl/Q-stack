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

    with open(fname, "rb") as f:
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
            def read_array(n, dtype):
                return np.frombuffer(f.read(dtype().itemsize * n), dtype=dtype)
            coefficients = read_array(nao*nao, np.float64).reshape(-1, nao)
            occupations  = read_array(nao,     np.float64)
            energies     = read_array(nao,     np.float64)
            irreps       = read_array(nao,     np.int32)
            cores        = read_array(nao,     np.int32)
            coefficients_ab.append(coefficients)
            energies_ab.append(energies)
            occupations_ab.append(occupations)

        coefficients_ab = np.array(coefficients_ab)
        energies_ab     = np.array(energies_ab)
        occupations_ab  = np.array(occupations_ab)

        return coefficients_ab, energies_ab, occupations_ab


def reorder_coeff_inplace(c_full, mol, reorder_dest='pyscf'):
    def _reorder_coeff(c):
        # In ORCA, def2-TZVP for metal is not sorted wrt angular momenta. TODO add the fixes
        for i in range(len(c)):
            c[:,i] = reorder_ao(mol, c[:,i], src='orca', dest=reorder_dest)
    [_reorder_coeff(c_full[i]) for i in range(c_full.shape[0])]


def read_gbw(mol, fname, reorder_dest='pyscf'):
    c, e, occ = _parse_gbw(fname)
    if reorder_dest is not None:
        reorder_coeff_inplace(c, mol, reorder_dest)
    return c, e, occ
