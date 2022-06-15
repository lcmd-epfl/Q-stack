from genericpath import isdir, isfile
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from numpy.lib.shape_base import split
import qml
import numpy as np
import os
from os.path import join
import random
import argparse
from qml.math import cho_solve
from qml.kernels import gaussian_kernel, laplacian_kernel

parser = argparse.ArgumentParser(description='This code is intended to produce atomic-SLATM representations starting from a compound-geometries folder with a LABELED list of atomic charges.')
parser.add_argument('--geometries', required=True, type=str, dest='geometries', help='The path to the folder containing the geometries .xyz files.')
parser.add_argument('--atom', required=False, type=str, dest='atom', help='The selected atom to produce the atomic-representation from the database.')
args = parser.parse_args()

def get_qml_compounds(dir, property_dict=None, geom_list=[]) :
    if len(geom_list) > 0 : compounds = [qml.Compound(xyz=f"{os.path.join(dir, f)}") for f in geom_list]
    else : compounds = [qml.Compound(xyz=f"{os.path.join(dir, f)}") for f in sorted(os.listdir(dir))]
    return compounds


def get_nuclear_charges(compound_list) :
    nuclear_charges = []
    for mol in compound_list :
        nuclear_charges.append(mol.nuclear_charges)
    return nuclear_charges

def parse_a_prop(compound) :
    elements = compound.atomtypes
    # print(elements)
    a_dict = dict()
    for i, e in enumerate(elements) :
        if e not in a_dict.keys() : a_dict[e] = [compound.properties[i]]
        else : a_dict[e].append(compound.properties[i])
    return a_dict


current_dir = os.getcwd()
geom_dir = join(current_dir, args.geometries)

molecules = get_qml_compounds(geom_dir)

z_charges = get_nuclear_charges(molecules)
slatm_basis = qml.representations.get_slatm_mbtypes(z_charges)

for i,mol in enumerate(molecules):
    print(i)
    mol.generate_slatm(slatm_basis, local=True)

if args.atom : atoms =[args.atom]
else : atoms = ['N', 'C', 'O', 'S', 'H']

a_slatm = dict()
for atom in atoms :
    a_slatm[atom] = []
    for mol in molecules :
        atoms_in = mol.atomtypes
        for i, a in enumerate(atoms_in) :
            if a == atom : a_slatm[atom].append(mol.representation[i])


for key in a_slatm :
    np.save(f'a_SLATM_{key}_QM7', a_slatm[key])
