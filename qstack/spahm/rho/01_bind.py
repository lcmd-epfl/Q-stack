import numpy as np
import sys
from os.path import join
from os import getcwd
import argparse


parser = argparse.ArgumentParser(description='This script generates a single file per atomic species binding all the feature-vector representations obtained using the single-structures representations generator script (1_DMbRep.py)')

parser.add_argument('--X-dir', required = True, type = str, dest='XDir', help = 'The path to the directory containing all the single-structure X representations.')
parser.add_argument('--species', required = False, type = str, nargs='+', dest='Species', help = 'The species contained in the DatBase.')

args = parser.parse_args()

# def main() :
cwd = getcwd()

list_ordered = np.loadtxt('list_structures.txt', dtype=str)

X_dir = join(cwd, args.XDir)

X_files = [join(X_dir, 'X_' + mol.split('.')[0] + '.npy') for mol in list_ordered]

# print(X_files)

if args.Species :
	species = list(sorted(args.Species))
else :
	species = sorted(["C", "H", "N", "O", "S"])
X_atom = dict()
for e in species :
	X_atom[e] = []
print(X_atom)
for f in X_files :
	X = np.load(f, allow_pickle=True)
	for a in X :
		X_atom[a[0]].append(a[1])
	print(f, "binded", sep=' ')

for e in X_atom.keys() :
	X_array = np.array(X_atom[e])
	name_out = 'X_' + e
	np .save(join(cwd, name_out), X_array)
	print('# Atoms for ', e, '=', X_array.shape, sep='\t')


# if __name__ == '__main__ ' : main()

