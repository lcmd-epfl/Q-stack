import os
from qstack import basis_opt

optimized_basis = basis_opt.opt.optimize_basis(["No"], ["../basis_opt/HH.bas"], ["rho/H2.xyz.rho_bond.npz"], gtol_in = 1e-7, method_in = "CG")

print(optimized_basis)
