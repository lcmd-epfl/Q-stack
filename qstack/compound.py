"""
Module containing all the operations to load, transform, and save molecular objects. 
"""

from pyscf import gto
from qstack import constants
from qstack.tools import rotate_euler


def xyz_to_mol(fin, basis, charge=0, spin=0):
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
    molxyz = '\n'.join(f.read().split('\n')[2:])
    f.close()

    # Define attributes to the Mole object and build it
    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    return mol

def mol_to_xyz(mol, fout, format='xyz'):
    """Converts a pyscf Mole object into a molecular file in xyz format.

    Args:
        pyscf Mole: pyscf Mole object.
        fout (str): name (including path) of the xyz file to write.

    Returns:
        pyscf Mole: pyscf Mole object.
    """

    format = format.lower()
    if format == 'xyz':
        coords = mol.atom_coords() * constants.BOHR2ANGS
        output = []
        if format == 'xyz':
            output.append('%d' % mol.natm)
            output.append('%d %d' % (mol.charge, mol.multiplicity))

        for i in range(mol.natm):
            symb = mol.atom_pure_symbol(i)
            x, y, z = coords[i]
            output.append('%-4s %14.5f %14.5f %14.5f' %
                          (symb, x, y, z))
        string = '\n'.join(output)

    else:
        raise NotImplementedError

    with open(fout, 'w') as f:
        f.write(string)
        f.write('\n')
    return string



def makeauxmol(mol, basis):
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

def rotate_molecule(mol, a, b, g, rad = False):
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
    rotated_coords = orig_coords@rotate_euler(a, b, g, rad) * constants.BOHR2ANGS
    atom_types = mol.elements

    rotated_mol = gto.Mole()
    rotated_mol.atom = list(zip(atom_types, rotated_coords.tolist()))
    rotated_mol.charge = mol.charge
    rotated_mol.spin = mol.spin
    rotated_mol.basis = mol.basis
    rotated_mol.build()

    return rotated_mol