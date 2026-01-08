"""Read Turbomole MOs files.

Tested with TURBOMOLE V7.1
"""

import re
import numpy as np
from qstack.mathutils.array import genfromtxt_var
from qstack.reorder import reorder_ao
from qstack.tools import Cursor


def read_mos(mol, fname, reorder_dest='pyscf'):
    """Read MOs from Turbomole `mos` file.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        fname (str): Path to the Turbomole MOs file.
        reorder_dest (str): Which AO ordering convention to use. Defaults to 'pyscf'.

    Note:
        Open-shell MOs files are not supported.

    Returns:
        c (ndarray): AO coefficients, shape (nao, nmo).
        e (ndarray): Orbital energies, shape (nmo,).
        title (str): Title of the MOs set ('$scfmo', '$uhfmo_alpha', or '$uhfmo_beta').

    Raises:
        RuntimeError: If the file format is invalid or unsupported.
    """
    with open(fname) as f:
        lines = [*filter(lambda x: not x.strip().startswith('#'), f.readlines())]

    title, _, fmt = lines[0].split()
    if title not in ['$scfmo', '$uhfmo_alpha', '$uhfmo_beta']:
        raise RuntimeError(f'Not a valid Turbomole MOs file or unsupported MO set ({title}). You can contribute!')

    re_int = r'(\d+)'
    re_flt = r'([0-].\d+[Ee][+-]\d+)'
    re_fmt = re.compile(f'format\\({re_int}d{re_int}\\.{re_int}\\)')
    re_eigen = re.compile(f'.*eigenvalue={re_flt}\\s+nsaos={re_int}')

    matcher = re_fmt.fullmatch(fmt)
    if matcher is None:
        raise RuntimeError('Cannot parse Turbomole format string')
    fmt = [*map(int, matcher.groups())]

    def read_c_vals(sl):
        c_lines = ''.join(lines[sl]).replace('D', 'E')
        return genfromtxt_var(c_lines, delimiter=fmt[1]).flatten()

    c = []
    e = []
    i = Cursor(i0=1, action='slicer')

    while True:
        matcher = re_eigen.match(lines[i.add(1).start].strip().replace('D', 'E'))
        if matcher is None:
            raise RuntimeError('Cannot parse eigenvalue line')
        nao, ei = int(matcher.group(2)), float(matcher.group(1))
        if mol.nao != nao:
            raise RuntimeError(f'Number of AOs mismatch {mol.nao} != {nao}')
        e.append(ei)

        ci = read_c_vals(i.add(nao // fmt[0]))
        if nao % fmt[0]:
            ci = np.hstack((ci, read_c_vals(i.add(1))))
        if len(ci)!=nao or np.isnan(ci).any():
            raise RuntimeError('AO coefficients size mismatch')
        c.append(ci)

        if lines[i.i]=='$end\n':
            break

    c, e = np.array(c).T, np.array(e)
    if reorder_dest is not None:
        idx, sign = reorder_ao(mol, None, src='turbomole', dest=reorder_dest)
        for i in range(c.shape[1]):
            c[:,i] = c[idx,i]*sign

    return c, e, title
