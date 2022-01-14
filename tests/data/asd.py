import qstack

mol = qstack.compound.xyz_to_mol("H2O.xyz", 'def2svp', charge =1, spin =1)
dm = qstack.density.dm.get_converged_dm(mol, 'pbe')
