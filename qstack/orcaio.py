import numpy as np
from qstack.math.matrix import from_tril
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
