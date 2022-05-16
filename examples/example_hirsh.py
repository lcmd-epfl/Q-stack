import numpy as np
import os
from qstack import compound, fields
from qstack.fields.decomposition import decompose
from qstack.fields.hirshfeld import hirshfeld_charges,spherical_atoms

path = os.path.dirname(os.path.realpath(__file__))
mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)

atm_bas = 'cc-pVQZ jkfit'
dm_atoms = spherical_atoms(set(mol.elements), atm_bas)

dm = fields.dm.get_converged_dm(mol,xc="pbe")
ho = hirshfeld_charges(mol, dm, dm_atoms=dm_atoms, atm_bas=atm_bas, dominant=True, occupations=False, grid_level=3)
print('dm ', ho)

auxmol, c = decompose(mol, dm, 'cc-pvqz jkfit')

ho = hirshfeld_charges(auxmol, c, dm_atoms=dm_atoms, atm_bas=atm_bas, dominant=True, occupations=False, grid_level=3)
print('fit', ho)
