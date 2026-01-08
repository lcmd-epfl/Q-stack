#!/usr/bin/env python3

import os
from qstack.io.cell2mol import get_cell, get_mol, get_ligand


def test_c2mio():
    path = os.path.dirname(os.path.realpath(__file__))
    cell = get_cell(f'{path}/data/cell2mol/YOXKUS.cif', workdir=f'{path}/data/cell2mol/')  # cell = get_cell('Cell_yoxkus.cell', workdir='.')
    # print(cell.moleclist)
    mol = get_mol(cell, mol_idx=0, ecp='def2-svp')
    assert mol.natm==52

    cell = get_cell(f'{path}/data/cell2mol/Cell_YOXKUS.cell', workdir='.')
    # print(cell.moleclist[0].ligands)
    mol_lig = get_ligand(cell, mol_idx=0, lig_idx=1)
    assert mol_lig.natm==47


if __name__ == '__main__':
    test_c2mio()
