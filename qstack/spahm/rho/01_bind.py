#!/usr/bin/env python3

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This script generates a single file per atomic species binding all the feature-vector representations obtained using the single-structures representations generator script (1_DMbRep.py)')
parser.add_argument('--X-dir',   required = True,                     type = str,            dest='XDir', help = 'The path to the directory containing all the single-structure X representations.')
parser.add_argument('--species', default = ["C", "H", "N", "O", "S"], type = str, nargs='+', dest='Species', help = 'The species contained in the DatBase.')
args = parser.parse_args()

species = list(sorted(args.Species))
X_files = sorted(os.listdir(args.XDir))

X_atom = {}
for e in species:
    X_atom[e] = []

for f in X_files :
    X = np.load(args.XDir+'/'+f, allow_pickle=True)
    for q,x in X :
        X_atom[q].append(x)
    print(f)

for e in X_atom.keys() :
    X_array = np.array(X_atom[e])
    name_out = 'X_' + e + '_' + args.XDir
    if name_out[-1] == '/': name_out = name_out[:-1]
    np.save(name_out, X_array)
    print('# Atoms for ', e, '=', X_array.shape)
