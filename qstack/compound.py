"""
Module containing all the operations to load, transform, and save molecular objects.
"""

import pickle
import numpy as np
from pyscf import gto, data
from qstack import constants
from qstack.tools import rotate_euler


def xyz_to_mol(fin, basis="def2-svp", charge=0, spin=0):
    """Reads a molecular file in xyz format and returns a pyscf Mole object.

    Args:
        fin (str): name (including path) of the xyz file to read.
        basis (str or dict): basis set.
        charge (int): charge of the molecule.
        spin (int): alpha electrons - beta electrons

    Returns:
        pyscf Mole: pyscf Mole object.
    """

    # Open and read the file
    f = open(fin, "r")
    molxyz = "\n".join(f.read().split("\n")[2:])
    f.close()

    # Define attributes to the Mole object and build it
    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    return mol


def mol_to_xyz(mol, fout, format="xyz"):
    """Converts a pyscf Mole object into a molecular file in xyz format.

    Args:
        pyscf Mole: pyscf Mole object.
        fout (str): name (including path) of the xyz file to write.

    Returns:
        pyscf Mole: pyscf Mole object.
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
    """Reads .

    Args:
        fin (str): name (including path) of the xyz file to read.
        basis (str or dict): basis set.
        charge (int): charge of the molecule.
        spin (int): alpha electrons - beta electrons

    Returns:
        pyscf Mole: pyscf Mole object.
    """

    from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
    from cell2mol.tmcharge_common import labels2formula

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


def make_auxmol(mol, basis):
    """Builds an auxiliary Mole object given a basis set and a pyscf Mole object.

    Args:
        mol (pyscf Mole): original pyscf Mole object.
        basis (str or dict): basis set.

    Returns:
        pyscf Mole: auxiliary pyscf Mole object.
    """

    # Define attributes to the auxiliary Mole object and build it
    auxmol = gto.Mole()
    auxmol.atom = mol.atom
    auxmol.charge = mol.charge
    auxmol.spin = mol.spin
    auxmol.basis = basis
    auxmol.build()

    return auxmol


def rotate_molecule(mol, a, b, g, rad=False):
    """Rotate a molecule: transform nuclear coordinates given a set of Euler angles.

    Args:
        mol (pyscf Mole): original pyscf Mole object.
        a (float): alpha Euler angle.
        b (float): beta Euler angle.
        g (float): gamma Euler angle.
        rad (bool) : Wheter the angles are in radians or not.


    Returns:
        pyscf Mole: Mole object with transformed coordinates.
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
    with open(frag_file, 'r') as f:
        fragments = [np.fromstring(line, dtype=int, sep=' ')-1 for line in f.readlines()]
    return fragments

def fragment_partitioning(fragments, prop_atom_inp, normalize=True):
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
