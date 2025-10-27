import sys
import os
import io
from cell2mol.read_write import load_binary
from cell2mol.unitcell import process_unitcell
from qstack.compound import xyz_to_mol


def get_cell2mol_xyz(mol):
    f = io.StringIO()
    sys.stdout, stdout = f, sys.stdout
    mol.print_xyz()
    xyz, sys.stdout = f.getvalue(), stdout
    f.close()
    return xyz, mol.totcharge, (mol.get_spin()-1 if hasattr(mol, 'get_spin') else 0)


def get_cell(fpath, workdir='.'):
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
    mol = cell.moleclist[mol_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp, read_string=True)


def get_ligand(cell, mol_idx=0, lig_idx=0, basis='minao', ecp=None):
    mol = cell.moleclist[mol_idx].ligands[lig_idx]
    xyz, charge, spin = get_cell2mol_xyz(mol)
    return xyz_to_mol(xyz, charge=charge, spin=spin, basis=basis, ecp=ecp, read_string=True)
