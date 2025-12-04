"""Molecular structure parsing and manipulation."""

import json
import re
import warnings
import numpy as np
from pyscf import gto, data
from qstack import constants
from qstack.reorder import get_mrange
from qstack.mathutils.array import stack_padding, loadtxt_var
from qstack.mathutils.rotation_matrix import rotate_euler
from qstack.tools import Cursor


# detects a charge-spin line, containing only two ints (one positive or negative, the other positive and nonzero)
_re_spincharge = re.compile(r'(?P<charge>[-+]?[0-9]+)\s+(?P<spinmult>[1-9][0-9]*)')
# fetches a single key=value or key:value pair, then matches a full line, for space-separated pairs
_re_singlekey = re.compile(r'\s*(?P<key>\w+)[=:](?P<val>[^\s,]+)\s*')
_re_keyline = re.compile(r'\s*(\w+[=:][^\s,]+\s+)*(\w+[=:][^\s,]+)\s*')
# fetches a single key=value or key:value pair, then matches a full line, for comma-separated pairs
_re_singlekey2 = re.compile(r'\s*(?P<key>\w+)\s*[=:]\s*(?P<val>[^\s,]+)\s*,?\s*')
_re_keyline2 = re.compile(r'{0}(,{0})*,?\s*'.format(r'\s*\w+\s*[=:]\s*[^\s,]+\s*'))
# matches an integer in any format python reads
_re_int = re.compile(r'[+-]?(?P<basisprefix>0[obxOBX])?[1-9a-fA-F][0-9a-fA-F]*')
# matches a floating-point number in any format python reads
_re_float = re.compile(r'[+-]?[0-9]*?([0-9]\.|\.[0-9]|[0-9])[0-9]*?([eEdD][+-]?[0-9]+)?')


def xyz_comment_line_parser(line):
    """Read the 'comment' line of a XYZ file and tries to infer its meaning.

    Args:
        line (str): Comment line from XYZ file.

    Returns:
        dict: Dictionary containing parsed properties (charge, spin, etc.).
    """
    line = line.strip()
    if line == '':
        return {}
    elif _re_spincharge.fullmatch(line):
        # possibility 1: the line only has charge and spin multiplicity
        # note: this skips the futher processing
        matcher = _re_spincharge.fullmatch(line)
        spinmult = int(matcher.group('spinmult'))
        charge = int(matcher.group('charge'))
        return {'charge':charge, 'spin':spinmult-1}
    elif _re_keyline.fullmatch(line):
        # possibility 2: space-separated key/value pairs
        line_parts = line.split()
        part_matching = _re_singlekey
        props = {}
    elif _re_keyline2.fullmatch(line):
        # possibility 3: comma-separated key/value pairs
        line_parts = line.split(',')
        part_matching = _re_singlekey2
        props = {}
    elif line.startswith('{'):
        # possibility 4: json
        line_parts = []
        try:
            props = json.loads(line.strip())
        except json.decoder.JSONDecodeError:
            return {}
    else:
        # other possibilities include having the name of the compound
        warnings.warn(f"could not interpret the data in the XYZ title line: {line}", RuntimeWarning, stacklevel=2)
        return {}

    for part in line_parts:
        part_matcher = part_matching.fullmatch(part)
        val = part_matcher.group('val')
        if val.lower() in ('f','no','false','off'):
            val = False
        elif val.lower() in ('t','yes','true','on'):
            val = True
        elif _re_int.fullmatch(val):
            prefix = _re_int.fullmatch(val).group('basisprefix')
            if prefix:
                val = int(val, basis=0)  # 'basis=0' means 'autodetect'
            else:
                val = int(val)
        elif _re_float.fullmatch(val):
            val = float(val)
        props[part_matcher.group('key')] = val

    if 'spin' in props:
        # we want a difference in electons (alpha-beta), but we expect the file to contain a spin multiplicity
        props['spin'] = props['spin']-1
    return props


def xyz_to_mol(inp, basis="def2-svp", charge=None, spin=None, ignore=False, unit=None, ecp=None, parse_comment=False):
    """Read a molecular file in xyz format and returns a pyscf Mole object.

    Args:
        inp (str): Path of the xyz file to read, or xyz file contents.
        basis (str or dict): Basis set. Defaults to "def2-svp".
        charge (int): Provide/override charge of the molecule. Defaults to None.
        spin (int): Provide/override spin of the molecule (alpha electrons - beta electrons). Defaults to None.
        ignore (bool): If True, assume molecule is closed-shell and assign charge either 0 or -1. Defaults to False.
        unit (str): Provide/override units (Ang or Bohr). Defaults to None.
        ecp (str): ECP to use. Defaults to None.
        parse_comment (bool): Whether to parse the comment line for properties. Defaults to False.

    Returns:
        pyscf.gto.Mole: pyscf Mole object containing the molecule information.

    Raises:
        RuntimeError: If units are not recognized or if minao basis requires ECP for heavy atoms.
    """
    if '\n' in inp:
        molxyz = gto.fromstring(inp)
    else:
        molxyz = gto.fromfile(inp)

    if parse_comment:
        if '\n' in inp:
            comment_line = inp.split('\n')[1]
        else:
            with open(inp) as f:
                _, comment_line = f.readline(), f.readline()
        props = xyz_comment_line_parser(comment_line)
    else:
        props = {}

    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    if ecp is not None:
        mol.ecp = ecp

    if unit is not None:
        pass
    elif 'unit' in props:
        unit = props['unit']
    else:
        unit = 'Angstrom'
    unit = unit.upper()[0]
    if unit not in ['B', 'A']:
        raise RuntimeError("Unknown units (use A[ngstrom] or B[ohr])")
    mol.unit = unit

    if ignore:
        if charge not in (0, None) or spin not in (0, None):
            warnings.warn("Spin and charge values are overwritten", RuntimeWarning, stacklevel=2)
        atoms = [int(q) if q.isdigit() else data.elements.ELEMENTS_PROTON[q] for q in loadtxt_var(molxyz, dtype='str', usecols=0)]
        mol.spin = 0
        mol.charge = -(sum(atoms)%2)
    else:
        if charge is not None:
            mol.charge = charge
        elif 'charge' in props:
            mol.charge = props['charge']
        else:
            mol.charge = 0

        if spin is not None:
            mol.spin = spin
        elif 'spin' in props:
            mol.spin = props['spin']
        else:
            mol.spin = 0

    mol.build()
    if mol.basis == 'minao' and ecp is None and (numbers(mol) > 36).any():
        msg = f"{mol.basis} basis set requires the use of effective core potentials for atoms with Z>36"
        raise RuntimeError(msg)
    return mol


def mol_to_xyz(mol, fout, fmt="xyz"):
    """Convert a pyscf Mole object into a molecular file in xyz format.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        fout (str): Name (including path) of the xyz file to write.
        fmt (str): Output format. Defaults to "xyz".

    Returns:
        str: String containing the xyz formatted data.

    Raises:
        NotImplementedError: If fmt is not "xyz".
    """
    fmt = fmt.lower()
    output = []
    if fmt == "xyz":
        coords = mol.atom_coords() * constants.BOHR2ANGS
        output.append(f"{mol.natm}\n{mol.charge} {mol.multiplicity}")
        output.extend([f"{mol.atom_pure_symbol(i):4s} {r[0]:14.5f} {r[1]:14.5f} {r[2]:14.5f}" for i, r in enumerate(coords)])
        output = "\n".join(output)
    else:
        raise NotImplementedError

    with open(fout, "w") as f:
        f.write(output+"\n")
    return output


def make_auxmol(mol, basis, copy_ecp=False):
    """Build an auxiliary Mole object given a basis set and a pyscf Mole object.

    Args:
        mol (pyscf.gto.Mole): Original pyscf Mole object.
        basis (str or dict): Basis set.
        copy_ecp (bool): Whether to copy ECP from original molecule. Defaults to False.

    Returns:
        pyscf.gto.Mole: Auxiliary pyscf Mole object.
    """
    auxmol = gto.Mole()
    auxmol.atom = mol.atom
    auxmol.charge = mol.charge
    auxmol.spin = mol.spin
    auxmol.basis = basis
    if copy_ecp:
        auxmol.ecp = mol.ecp
    auxmol.build()
    return auxmol


def rotate_molecule(mol, a, b, g, rad=False):
    """Rotate a molecule: transform nuclear coordinates given a set of Cardan angles.

    Args:
        mol (pyscf.gto.Mole): Original pyscf Mole object.
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool): Whether the angles are in radians. Defaults to False (degrees).

    Returns:
        pyscf.gto.Mole: pyscf Mole object with transformed coordinates.
    """
    rotated_coords = mol.atom_coords() @ rotate_euler(a, b, g, rad) * constants.BOHR2ANGS
    rotated_mol = gto.Mole()
    rotated_mol.atom = [*zip(mol.elements, rotated_coords, strict=True)]
    rotated_mol.charge = mol.charge
    rotated_mol.spin = mol.spin
    rotated_mol.basis = mol.basis
    rotated_mol.ecp = mol.ecp
    rotated_mol.build()
    return rotated_mol


def fragments_read(frag_file):
    """Load fragment definition from a file.

    Args:
        frag_file (str): Path to the fragment file containing space-separated atom indices (1-based).

    Returns:
        list: List of numpy arrays containing the fragment indices.
    """
    with open(frag_file) as f:
        fragments = [np.fromstring(line, dtype=int, sep=' ')-1 for line in f]
    return fragments


def fragment_partitioning(fragments, prop_atom_inp, normalize=True):
    """Compute the contribution of each fragment.

    Args:
        fragments (list): Fragment definition as list of numpy arrays.
        prop_atom_inp (numpy.ndarray or list of numpy.ndarray): Atomic contributions to property(ies).
        normalize (bool): Whether to normalize fragment partitioning. Defaults to True.

    Returns:
        list or numpy.ndarray: Contribution of each fragment. Returns list if input was list, array otherwise.
    """
    props_atom = prop_atom_inp if type(prop_atom_inp) is list else [prop_atom_inp]

    props_frag = []
    for prop_atom in props_atom:
        prop_frag = np.array([prop_atom[k].sum() for i, k in enumerate(fragments)])
        if normalize:
            prop_frag *= 100.0 / prop_frag.sum()
        props_frag.append(prop_frag)

    return props_frag if type(prop_atom_inp) is list else props_frag[0]


def make_atom(q, basis, ecp=None):
    """Create a single-atom molecule at the origin.

    Args:
        q (str): Element symbol.
        basis (str or dict): Basis set.
        ecp (str): ECP to use. Defaults to None.

    Returns:
        pyscf.gto.Mole: Single-atom pyscf Mole object.
    """
    mol = gto.Mole()
    mol.atom = q + " 0.0 0.0 0.0"
    mol.charge = 0
    mol.spin = data.elements.ELEMENTS_PROTON[q] % 2
    mol.basis = basis
    if ecp is not None:
        mol.ecp = ecp
    mol.build()
    return mol


def basis_flatten(mol, return_both=True, return_shells=False):
    """Flatten a basis set definition for AOs.

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.
        return_both (bool): Whether to return both AO info and primitive Gaussian info. Defaults to True.
        return_shells (bool): Whether to return angular momenta and starting indices per shell
            (2l+1-sized group of orbitals that share a radial function and l value). Defaults to False.

    Returns:
        - numpy.ndarray: 3×mol.nao int array where each column corresponds to an AO and rows are:
            - 0: atom index,
            - 1: angular momentum quantum number l,
            - 2: magnetic quantum number m.
        If return_both is True, also returns:
        - numpy.ndarray: 2×mol.nao×max_n float array where index (i,j,k) means:
            - i: 0 for exponent, 1 for contraction coefficient of a primitive Gaussian,
            - j: AO index,
            - k: radial function index (padded with zeros if necessary).
        If return_shells is True, also returns:
        - numpy.ndarray: starting AO indices for each shell.
    """
    x = []
    ao_starts = []
    cursor = Cursor(action='ranger')
    y = np.zeros((3, mol.nao), dtype=int)
    i = Cursor(action='slicer')
    a = mol.bas_exps()
    for iat in range(mol.natm):
        for bas_id in mol.atom_shell_ids(iat):
            l = mol.bas_angular(bas_id)
            n = mol.bas_nctr(bas_id)
            cs = mol.bas_ctr_coeff(bas_id)
            msize = 2*l+1
            if return_both:
                for c in cs.T:
                    ac = np.array([a[bas_id], c])
                    x.extend([ac]*msize)
            y[:,i(msize*n)] = np.vstack((np.array([[iat, l]]*msize*n).T, [*get_mrange(l)]*n))
            if return_shells:
                ao_starts.extend(cursor.add(msize*n)[::msize])

    ret = [y]
    if return_both:
        ret.append(stack_padding(x).transpose((1,0,2)))
    if return_shells:
        ret.append(np.array(ao_starts))
    return ret[0] if len(ret)==1 else ret


def numbers(mol):
    """Get atom numbers of a molecule.

    Use this function to get atomic NUMBERS to index elements.
    Use `mol.atom_charges()` to get CHARGES (it returns effective charges when ECP are used).

    Args:
        mol (pyscf.gto.Mole): pyscf Mole object.

    Returns:
        numpy.ndarray: Array of atomic numbers.
    """
    return np.array([data.elements.charge(q) for q in mol.elements])
