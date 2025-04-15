"""
Module containing all the operations to load, transform, and save molecular objects.
"""

import json, re
import pickle
import warnings
import numpy as np
from pyscf import gto, data
from qstack import constants
from qstack.tools import rotate_euler


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
    """reads the 'comment' line of a XYZ file, and tries to infer its meaning"""
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
        print("warning: could not interpret the data in the XYZ title line:", line)
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

def xyz_to_mol(fin, basis="def2-svp", charge=None, spin=None, ignore=False, unit=None, ecp=None):
    """Reads a molecular file in xyz format and returns a pyscf Mole object.

    Args:
        fin (str): Name (including path) of the xyz file to read.
        basis (str or dict): Basis set.
        charge (int): Provide/override charge of the molecule.
        spin (int): Provide/override spin of the molecule (alpha electrons - beta electrons).
        ignore (bool): If assume molecule closed-shell an assign charge either 0 or -1
        unit (str): Provide/override units (Ang or Bohr)
        ecp (str) : ECP to use

    Returns:
        A pyscf Mole object containing the molecule information.
    """

    # Open and read the file
    molxyz = gto.fromfile(fin)
    with open(fin, "r") as f:
        _ = f.readline()
        comment_line = f.readline()
        props = xyz_comment_line_parser(comment_line)

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
            warnings.warn("Spin and charge values are overwritten", RuntimeWarning)
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


def mol_to_xyz(mol, fout, format="xyz"):
    """Converts a pyscf Mole object into a molecular file in xyz format.

    Args:
        pyscf Mole: pyscf Mole object.
        fout (str): Name (including path) of the xyz file to write.

    Returns:
        A file in xyz format containing the charge, total spin and molecular coordinates.
    """

    format = format.lower()
    if format == "xyz":
        coords = mol.atom_coords() * constants.BOHR2ANGS
        output = []
        if format == "xyz":
            output.append("%d" % mol.natm)
            output.append("%d %d" % (mol.charge, mol.multiplicity))

        for i in range(mol.natm):
            symb = mol.atom_pure_symbol(i)
            x, y, z = coords[i]
            output.append("%-4s %14.5f %14.5f %14.5f" % (symb, x, y, z))
        string = "\n".join(output)

    else:
        raise NotImplementedError

    with open(fout, "w") as f:
        f.write(string)
        f.write("\n")
    return string


def gmol_to_mol(fin, basis="def2-svp"):
    """Reads a molecular file in gmol format and returns a pyscf Mole object.

    Args:
        fin (str): Name (including path) of the xyz file to read.
        basis (str or dict): Basis set.
        charge (int): Charge of the molecule.
        spin (int): Alpha electrons - beta electrons.

    Returns:
        pyscf Mole: pyscf Mole object.
    """

    try:
        from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
        from cell2mol.tmcharge_common import labels2formula
    except ImportError:
            print("""

ERROR: cannot import cell2mol. Have you installed qstack with the \gmol\" option?\n\n
(for instance, with `pip install qstack[gmol] or `pip install qstack[all]``)

""")
    raise


    # Open and read the file
    if fin.endswith(".gmol"):

        with open(fin, "rb") as pickle_file:
            gmol = pickle.load(pickle_file)

            if hasattr(gmol, "cellvec"):
                gmoltype = "cell"
            elif gmol.type == "Ligand":
                gmoltype = "ligand"
            elif gmol.type == "Other" or gmol.type == "Complex":
                gmoltype = "molecule"

            refcode = gmol.refcode

            if gmoltype == "cell":
                cell = gmol
                for mol in cell.moleclist:
                    if not hasattr(mol, "totcharge"):
                        print(
                            "Total Charge is Missing for molecule:",
                            refcode,
                            mol.type,
                            mol.natoms,
                            mol.labels,
                        )
                    else:
                        print(
                            f"Info (Charge, number of atoms): {mol.totcharge}, {mol.natoms}"
                        )

            elif gmoltype == "ligand":
                lig = gmol
                if not hasattr(lig, "totcharge"):
                    print(
                        "Total Charge is Missing for Ligand:",
                        refcode,
                        lig.type,
                        lig.natoms,
                    )
                elif not hasattr(lig, "totmconnec"):
                    print(
                        "ML connectivity is Missing for Ligand:",
                        refcode,
                        lig.type,
                        lig.natoms,
                    )
                else:
                    print(
                        f"Info (Charge, number of atoms, denticity): {lig.totcharge}, {lig.totmconnec}"
                    )

            elif gmoltype == "molecule":
                mol = gmol
                if not hasattr(mol, "totcharge"):
                    print(
                        "Total Charge is Missing for Molecule:",
                        refcode,
                        mol.type,
                        mol.natoms,
                        mol.labels,
                    )
                else:
                    print(
                        f"Info (Charge, number of atoms): {mol.totcharge}, {mol.natoms}"
                    )

    # Define attributes to the Mole object and build it
    mole = gto.Mole()
    atoms = list(zip(mol.labels, mol.coord))
    mole.atom = atoms
    mole.basis = basis
    mole.charge = mol.totcharge
    mole.spin = (sum(mol.atnums) - mol.totcharge) % 2
    mole.build()

    return mole


def make_auxmol(mol, basis, copy_ecp=False):
    """Builds an auxiliary Mole object given a basis set and a pyscf Mole object.

    Args:
        mol (pyscf Mole): Original pyscf Mole object.
        basis (str or dict): Basis set.

    Returns:
        An auxiliary pyscf Mole object.
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
        mol (pyscf Mole): Original pyscf Mole object.
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool) : Wheter the angles are in radians or not.


    Returns:
        A pyscf Mole object with transformed coordinates.
    """

    orig_coords = mol.atom_coords()
    rotated_coords = orig_coords @ rotate_euler(a, b, g, rad) * constants.BOHR2ANGS
    atom_types = mol.elements

    rotated_mol = gto.Mole()
    rotated_mol.atom = list(zip(atom_types, rotated_coords.tolist()))
    rotated_mol.charge = mol.charge
    rotated_mol.spin = mol.spin
    rotated_mol.basis = mol.basis
    rotated_mol.build()

    return rotated_mol



def fragments_read(frag_file):
    """Loads fragement definition from a frag file.

    Args:
        frag_file (str): Name (including path) of the frag file to read.

    Returns:
        A list of arrays containing the fragments.
    """
    with open(frag_file, 'r') as f:
        fragments = [np.fromstring(line, dtype=int, sep=' ')-1 for line in f.readlines()]
    return fragments

def fragment_partitioning(fragments, prop_atom_inp, normalize=True):
    """Computes the contribution of each fragment.

    Args:
        fragments (numpy ndarray): Fragment definition
        prop_atom_inp (list of arrays or array): Coefficients densities.
        normalize (bool): Normalized fragment partitioning. Defaults to True.

    Returns:
        A list of arrays or an array containing the contribution of each fragment.
    """

    if type(prop_atom_inp)==list:
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

    if type(prop_atom_inp)==list:
        return props_frag
    else:
        return props_frag[0]


def make_atom(q, basis):
    mol = gto.Mole()
    mol.atom = q + " 0.0 0.0 0.0"
    mol.charge = 0
    mol.spin = data.elements.ELEMENTS_PROTON[q] % 2
    mol.basis = basis
    mol.build()
    return mol

def singleatom_basis_enumerator(basis):
    """Enumerates the different tensors of atomic orbitals within a 1-atom basis set
    Each tensor is a $2l+2$-sized group of orbitals that share a radial function and $l$ value.
    For each tensor, return the values of $l$, $n$ (an arbitrary radial-function counter that starts at 0),
    as well as AO range
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

