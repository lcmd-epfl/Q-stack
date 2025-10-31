"""
Module containing all the operations to load, transform, and save molecular objects.
"""

import json
import re
import warnings
import numpy as np
from pyscf import gto, data
from qstack import constants
from qstack.mathutils.rotation_matrix import rotate_euler



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
    """Reads the 'comment' line of a XYZ file and tries to infer its meaning.

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
        matcher = _re_spincharge.fullmatch(line)
        spinmult = int(matcher.group('spinmult'))
        charge = int(matcher.group('charge'))
        # note: this skips the futher processing
        return {'charge':charge, 'spin':spinmult-1}
    elif _re_keyline.fullmatch(line):
        # possibility 2: space-separated key/value pairs
        line_parts = line.split()  # split across any whitespace
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

def xyz_to_mol(inp, basis="def2-svp", charge=None, spin=None, ignore=False, unit=None, ecp=None, parse_comment=False, read_string=False):
    """Reads a molecular file in xyz format and returns a pyscf Mole object.

    Args:
        inp (str): Path of the xyz file to read, or xyz file contents if read_string==True.
        basis (str or dict): Basis set. Defaults to "def2-svp".
        charge (int): Provide/override charge of the molecule. Defaults to None.
        spin (int): Provide/override spin of the molecule (alpha electrons - beta electrons). Defaults to None.
        ignore (bool): If True, assume molecule is closed-shell and assign charge either 0 or -1. Defaults to False.
        unit (str): Provide/override units (Ang or Bohr). Defaults to None.
        ecp (str): ECP to use. Defaults to None.
        parse_comment (bool): Whether to parse the comment line for properties. Defaults to False.
        read_string (bool): Whether inp is a string containing xyz data rather than a file path. Defaults to False.

    Returns:
        pyscf.gto.Mole: pyscf Mole object containing the molecule information.

    Raises:
        RuntimeError: If units are not recognized or if minao basis requires ECP for heavy atoms.
    """

    if read_string:
        molxyz = gto.fromstring(inp)
    else:
        molxyz = gto.fromfile(inp)

    if parse_comment:
        if read_string:
            comment_line = inp.split('\n')[1]
        else:
            with open(inp) as f:
                _ = f.readline()
                comment_line = f.readline()
        props = xyz_comment_line_parser(comment_line)
    else:
        props = [None]

    # Define attributes to the Mole object and build it
    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis

    # Check the units for the pyscf driver
    if unit is not None:
        pass
    elif 'unit' in props:
        unit = props['unit']
    else:
        unit = 'Angstrom'
    unit = unit.upper()[0]
    if unit not in ['B', 'A']:
        raise RuntimeError("Unknown units (use Ängstrom or Bohr)")
    mol.unit = unit

    if ignore:
        if charge not in (0, None) or spin not in (0, None):
            warnings.warn("Spin and charge values are overwritten", RuntimeWarning, stacklevel=2)
        mol.spin = 0
        mol.charge = - sum(mol.atom_charges())%2
    else:
        if charge is not None:
            mol.charge = charge
        elif 'charge' in props:
            mol.charge = props['charge']
        else:
            # no ignore, no charge/spin specified:
            # let's hope we have a set of neutral, closed shell compounds!
            mol.charge = 0

        if spin is not None:
            mol.spin = spin
        elif 'spin' in props:
            mol.spin = props['spin']
        else:
            mol.spin = 0

    if ecp is not None:
        mol.ecp = ecp

    mol.build()
    species_charges = [data.elements.charge(z) for z in mol.elements]
    if mol.basis == 'minao' and ecp is None and (np.array(species_charges) > 36).any():
        msg = f"{mol.basis} basis set requires the use of effective core potentials for atoms with Z>36"
        raise RuntimeError(msg)
    return mol


def mol_to_xyz(mol, fout, fmt="xyz"):
    """Converts a pyscf Mole object into a molecular file in xyz format.

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
    if fmt == "xyz":
        coords = mol.atom_coords() * constants.BOHR2ANGS
        output = []
        if fmt == "xyz":
            output.append(str(mol.natm))
            output.append(f"{mol.charge} {mol.multiplicity}")

        for i in range(mol.natm):
            symb = mol.atom_pure_symbol(i)
            x, y, z = coords[i]
            output.append(f"{symb:4s} {x:14.5f} {y:14.5f} {z:14.5f}")
        string = "\n".join(output)

    else:
        raise NotImplementedError

    with open(fout, "w") as f:
        f.write(string)
        f.write("\n")
    return string


def make_auxmol(mol, basis, copy_ecp=False):
    """Builds an auxiliary Mole object given a basis set and a pyscf Mole object.

    Args:
        mol (pyscf.gto.Mole): Original pyscf Mole object.
        basis (str or dict): Basis set.
        copy_ecp (bool): Whether to copy ECP from original molecule. Defaults to False.

    Returns:
        pyscf.gto.Mole: Auxiliary pyscf Mole object.
    """

    # Define attributes to the auxiliary Mole object and build it
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
    """Rotate a molecule: transform nuclear coordinates given a set of Euler angles.

    Args:
        mol (pyscf.gto.Mole): Original pyscf Mole object.
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool): Whether the angles are in radians. Defaults to False (degrees).

    Returns:
        pyscf.gto.Mole: pyscf Mole object with transformed coordinates.
    """

    orig_coords = mol.atom_coords()
    rotated_coords = orig_coords @ rotate_euler(a, b, g, rad) * constants.BOHR2ANGS
    atom_types = mol.elements

    rotated_mol = gto.Mole()
    rotated_mol.atom = list(zip(atom_types, rotated_coords.tolist(), strict=True))
    rotated_mol.charge = mol.charge
    rotated_mol.spin = mol.spin
    rotated_mol.basis = mol.basis
    rotated_mol.build()

    return rotated_mol



def fragments_read(frag_file):
    """Loads fragment definition from a frag file.

    Args:
        frag_file (str): Name (including path) of the frag file to read.

    Returns:
        list: List of numpy arrays containing the fragment indices.
    """
    with open(frag_file) as f:
        fragments = [np.fromstring(line, dtype=int, sep=' ')-1 for line in f]
    return fragments

def fragment_partitioning(fragments, prop_atom_inp, normalize=True):
    """Computes the contribution of each fragment.

    Args:
        fragments (list): Fragment definition as list of numpy arrays.
        prop_atom_inp (list or numpy.ndarray): Coefficients densities, either as list of arrays or single array.
        normalize (bool): Whether to normalize fragment partitioning. Defaults to True.

    Returns:
        list or numpy.ndarray: Contribution of each fragment. Returns list if input was list, array otherwise.
    """

    if type(prop_atom_inp) is list:
        props_atom = prop_atom_inp
    else:
        props_atom = [prop_atom_inp]

    props_frag = []
    for prop_atom in props_atom:
        prop_frag = np.zeros(len(fragments))
        for i, k in enumerate(fragments):
            prop_frag[i] = prop_atom[k].sum()
            prop_frag[i] = prop_atom[k].sum()
        props_frag.append(prop_frag)

    if normalize:
        for i, prop_frag in enumerate(props_frag):
            tot = prop_frag.sum()
            props_frag[i] *= 100.0 / tot

    if type(prop_atom_inp) is list:
        return props_frag
    else:
        return props_frag[0]


def make_atom(q, basis):
    """Create a single-atom molecule at the origin.

    Args:
        q (str): Element symbol.
        basis (str or dict): Basis set.

    Returns:
        pyscf.gto.Mole: Single-atom pyscf Mole object.
    """
    mol = gto.Mole()
    mol.atom = q + " 0.0 0.0 0.0"
    mol.charge = 0
    mol.spin = data.elements.ELEMENTS_PROTON[q] % 2
    mol.basis = basis
    mol.build()
    return mol

def singleatom_basis_enumerator(basis):
    """Enumerates the different tensors of atomic orbitals within a 1-atom basis set.

    Each tensor is a 2l+1-sized group of orbitals that share a radial function and l value.

    Args:
        basis (list): Basis set definition in pyscf format.

    Returns:
        tuple: A tuple containing:
            - l_per_bas (list): Angular momentum quantum number l for each basis function.
            - n_per_bas (list): Radial function counter n (starting at 0) for each basis function.
            - ao_starts (list): Starting index in AO array for each basis function.
    """
    ao_starts = []
    l_per_bas = []
    n_per_bas = []
    cursor = 0
    cursor_per_l = []
    for bas in basis:
        # shape of `bas`, l, then another optional constant, then lists [exp, coeff, coeff, coeff]
        # that make a matrix between the number of functions (number of coeff per list)
        # and the number of primitive gaussians (one per list)
        l = bas[0]
        while len(cursor_per_l) <= l:
            cursor_per_l.append(0)

        n_count = len(bas[-1])-1
        n_start = cursor_per_l[l]
        cursor_per_l[l] += n_count

        l_per_bas += [l] * n_count
        n_per_bas.extend(range(n_start, n_start+n_count))
        msize = 2*l+1
        ao_starts.extend(range(cursor, cursor+msize*n_count, msize))
        cursor += msize*n_count
    return l_per_bas, n_per_bas, ao_starts

