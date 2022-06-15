#!/usr/bin/env python3
# module purge; module load intel/19.0.4 anaconda/5.2.0/python-3.6
import os
import numpy as np
import argparse
import qml

parser = argparse.ArgumentParser(description='This script produces aSLATM representations for molecules in a given directory.')
parser.add_argument('--geometries', required=True, type=str, dest='geometries', help='The path to the folder containing the geometries .xyz files.')
args = parser.parse_args()

def main():
    molecules = [qml.Compound(xyz=f"{os.path.join(args.geometries, f)}") for f in sorted(os.listdir(args.geometries))]
    charges   = [mol.nuclear_charges for mol in molecules]
    mbtypes   = qml.representations.get_slatm_mbtypes(charges)

    for i,mol in enumerate(molecules):
        print(i)
        mol.generate_slatm(mbtypes, local=True)

    elements = set(np.hstack([np.array(mol.atomtypes) for mol in molecules]))
    a_slatm  = dict()
    for q in elements :
        a_slatm[q] = []
    for mol in molecules :
        for q,v in zip(mol.atomtypes, mol.representation) :
            a_slatm[q].append(v)

    for q in a_slatm :
        np.save(f'a_SLATM_{q}_QM7', a_slatm[q])

if __name__ == '__main__':
    main()
