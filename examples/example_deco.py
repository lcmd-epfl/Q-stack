import os
import numpy as np
from qstack import compound, fields
from qstack.fields.decomposition import decompose, correct_N, correct_N_atomic
from qstack.fields.density2file import coeffs_to_cube, coeffs_to_molden

path = os.path.dirname(os.path.realpath(__file__))
mol  = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
dm   = fields.dm.get_converged_dm(mol,xc="pbe")
auxmol, c = decompose(mol, dm, 'cc-pvqz jkfit')
print("Expansion Coefficients:", c)

N = fields.decomposition.number_of_electrons_deco(auxmol, c)

print("Number of electrons after decomposition: ", N)

coeffs_to_cube(auxmol, c, 'H2O.cube')
print('density saved to H2O.cube')

coeffs_to_molden(auxmol, c, 'H2O.molden')
print('density saved to H2O.molden')

c = correct_N(auxmol, c)
N = fields.decomposition.number_of_electrons_deco(auxmol, c)
print(N)


c = correct_N_atomic(auxmol, np.array([8,1,1]), c)
