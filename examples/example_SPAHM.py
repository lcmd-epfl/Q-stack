import os
from qstack import compound, spahm

path = os.path.dirname(os.path.realpath(__file__))
mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)
X1 = spahm.compute_spahm.get_spahm_representation(mol, "lb-hfs")

mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=1, spin=1)
X2 = spahm.compute_spahm.get_spahm_representation(mol, "lb-hfs")

print(X1)
print(X2)
