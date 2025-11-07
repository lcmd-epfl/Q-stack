"""Converter from cell2mol Cell objects to PySCF Mole."""

import sys
import os
import io
from cell2mol.read_write import load_binary
from cell2mol.unitcell import process_unitcell
from qstack.compound import xyz_to_mol


def get_cell2mol_xyz(mol):
    """Extract XYZ coordinates, charge, and spin from a cell2mol object.

    Args:
        mol: cell2mol molecule or ligand object.

    Returns:
        tuple: A tuple containing:
        - xyz (str): XYZ coordinate string.
        - charge (int): Total charge of the molecule.
        - spin (int): Number of unpaired electrons of the molecule (multiplicity - 1)
        for molecules and None for ligands.
    """
    f = io.StringIO()
    sys.stdout, stdout = f, sys.stdout
    mol.print_xyz()
    xyz, sys.stdout = f.getvalue(), stdout
    f.close()
    return xyz, mol.totcharge, (mol.get_spin()-1 if hasattr(mol, 'get_spin') else None)


def get_cell(fpath, workdir='.'):
    """Load a unit cell from a .cell or .cif file.

    If a .cif file is provided, the function checks for a corresponding .cell file
    in the working directory. If it exists, it loads the .cell file; otherwise, it
    calls cell2mol to process the .cif file to generate the unit cell.

    Args:
        fpath (str): Path to the input file (.cell or .cif).
        workdir (str): Directory to read / write .cell file and logs if a .cif file
            is provided. Defaults to '.'.

    Returns:
        cell2mol.unitcell: Unit cell object.

    Raises:
        NotImplementedError: If the file extension is not .cell or .cif.
    """
    ext = os.path.splitext(fpath)[-1]
    if ext=='.cell':
        cell = load_binary(fpath)
    elif ext=='.cif':
        name = os.path.basename(os.path.splitext(fpath)[0])
        cell_path = f'{workdir}/Cell_{name}.cell'
        if os.path.isfile(cell_path):
            cell = load_binary(cell_path)
        else:
            cell = process_unitcell(fpath, name, workdir, cif_bond_info=True, debug=0)
    else:
        raise NotImplementedError(f'{ext} input is not supported')
    return cell


def get_mol(cell, mol_idx=0, basis='minao', ecp=None):
    """Convert a molecule in a cell2mol unit cell object to a pyscf Mole object.

    Args:
        cell: cell2mol unit cell object.
        mol_idx (int): Index of the molecule in the cell. Defaults to 0.
        basis (str or dict): Basis set. Defaults to 'minao'.
        ecp (str): Effective core potential. Defaults to None.

    Returns:
        pyscf.gto.Mole: pyscf Mole object for the molecule.
    """
    mol = cell.moleclist[mol_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp)


def get_ligand(cell, mol_idx=0, lig_idx=0, basis='minao', ecp=None):
    """Convert a ligand in a cell2mol unit cell object to a pyscf Mole object.

    Args:
        cell: cell2mol unit cell object.
        mol_idx (int): Index of the molecule in the cell. Defaults to 0.
        lig_idx (int): Index of the ligand in the molecule. Defaults to 0.
        basis (str or dict): Basis set. Defaults to 'minao'.
        ecp (str): Effective core potential. Defaults to None.

    Returns:
        pyscf.gto.Mole: pyscf Mole object for the ligand.
    """
    mol = cell.moleclist[mol_idx].ligands[lig_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp)
