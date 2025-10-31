import sys
import os
import io
from cell2mol.read_write import load_binary
from cell2mol.unitcell import process_unitcell
from qstack.compound import xyz_to_mol


def get_cell2mol_xyz(mol):
    """Extract XYZ coordinates, charge, and spin from a cell2mol molecule object.

    Args:
        mol: cell2mol molecule object.

    Returns:
        tuple: A tuple containing:
            - xyz (str): XYZ coordinate string.
            - charge (int): Total charge of the molecule.
            - spin (int): Spin of the molecule (alpha electrons - beta electrons).
    """
    f = io.StringIO()
    sys.stdout, stdout = f, sys.stdout
    mol.print_xyz()
    xyz, sys.stdout = f.getvalue(), stdout
    f.close()
    return xyz, mol.totcharge, (mol.get_spin()-1 if hasattr(mol, 'get_spin') else 0)


def get_cell(fpath, workdir='.'):
    """Load a unit cell from a .cell or .cif file.

    Args:
        fpath (str): Path to the input file (.cell or .cif).
        workdir (str): Working directory for temporary files. Defaults to '.'.

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
    """Extract a pyscf Mole object from a cell2mol unit cell.

    Args:
        cell: cell2mol unit cell object.
        mol_idx (int): Index of the molecule in the cell. Defaults to 0.
        basis (str or dict): Basis set. Defaults to 'minao'.
        ecp (str): Effective core potential. Defaults to None.

    Returns:
        pyscf.gto.Mole: pyscf Mole object containing the molecule information.
    """
    mol = cell.moleclist[mol_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp, read_string=True)


def get_ligand(cell, mol_idx=0, lig_idx=0, basis='minao', ecp=None):
    """Extract a ligand as a pyscf Mole object from a cell2mol unit cell.

    Args:
        cell: cell2mol unit cell object.
        mol_idx (int): Index of the molecule in the cell. Defaults to 0.
        lig_idx (int): Index of the ligand. Defaults to 0.
        basis (str or dict): Basis set. Defaults to 'minao'.
        ecp (str): Effective core potential. Defaults to None.

    Returns:
        pyscf.gto.Mole: pyscf Mole object containing the ligand information.
    """
    mol = cell.moleclist[mol_idx].ligands[lig_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp, read_string=True)
