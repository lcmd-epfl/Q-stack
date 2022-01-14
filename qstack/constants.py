'''
NIST physical constants and unit conversion

https://physics.nist.gov/cuu/Constants/
https://physics.nist.gov/cuu/Constants/Table/allascii.txt
'''

# Constants
SPEED_LIGHT = 299792458.0
PLANCK = 6.62607004e-34
HBAR = PLANCK/(2*3.141592653589793) 
FUND_CHARGE = 1.6021766208e-19
MOL_NA = 6.022140857e23       
MASS_E = 9.10938356e-31
MASS_P = 1.672621898e-27
ATOMIC_MASS = 1e-3/MOL_NA
DEBYE = 3.335641e-30
BOLTZMANN = 1.38064852e-23      # J/K 

# Conversion
BOHR2ANGS = 0.52917721092  # Angstroms
HARTREE2J = HBAR**2/(MASS_E*(BOHR2ANGS*1e-10)**2)
HARTREE2EV = 27.21138602
AU2DEBYE = FUND_CHARGE * BOHR2ANGS*1e-10 / DEBYE # 2.541746
