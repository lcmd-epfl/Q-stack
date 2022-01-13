import numpy as np
from pyscf import gto

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
    auxmol.basis = basis
    auxmol.build()
    
    return auxmol
