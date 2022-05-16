import os
from qstack import compound, fields,basis_opt

def test_hf_otpd():

    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'def2svp', charge=0, spin=0)

    dm = fields.dm.get_converged_dm(mol,xc="pbe")
    otpd, grid = fields.hf_otpd.hf_otpd(mol, dm, return_all = True )

    opt_dict = {'atom':mol.atom, 'rho': otpd, 'coords': grid.coords , 'weights': grid.weights}
    optimized_basis = basis_opt.opt.optimize_basis(["O"], [mol._basis], [opt_dict], gtol_in = 1e-7, method_in = "CG")

    assert(isinstance(optimized_basis, dict))
