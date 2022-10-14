import os
import sys
import numpy as np
import qtsack

import argparse
parser = argparse.ArgumentParser(description='Script to regresse global molecular properties from atomic environnments descriptor (RHO-SPAHM).')
parser.add_argument('--x-dir',  type=str,   dest = 'Xdir',      required = True,                            help = 'the directory locating all atomic representation per molecular-file (.npy)')
parser.add_argument('--debug',              dest = 'DEBUG',     required = False,   action = 'store_true',   help = 'the directory locating all atomic representation per molecular-file (.npy)')

args = parser.parse_args()


DEBUG = args.DEBUG

def mol_to_dict(mol, species):
    mol_dict = {s:[]  for s in species}
    for a in mol:
        mol_dict[a[0]].append(a[1])
    for k in mol_dict.keys():
        mol_dict[k] = np.array(mol_dict[k])
    return mol_dict


def get_covariance(mol1, mol2, max_sizes):
    species = sorted(max_sizes.keys())
    mol1_dict = mol_to_dict(mol1, species)
    mol2_dict = mol_to_dict(mol2, species)
    max_size = sum(max_sizes.values())
    K_covar = np.zeros((max_size, max_size))
    idx = 0
    for s in species:
        n1 = len(mol1_dict[s])
        n2 = len(mol2_dict[s])
        if n1 == 0 or n2 == 0:
            idx += s_size
            continue
        s_size = max_sizes[s]
        x1 = np.pad(mol1_dict[s], ((0, s_size - n1),(0,0)), 'constant')
        x2 = np.pad(mol2_dict[s], ((0, s_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+s_size, idx:idx+s_size] = kernel(x1, x2)
        idx += s_size
    return K_covar

def avg_kernel(kernel):
    avg = np.sum(kernel) / len(kernel)**2
    return avg


def get_global_K(mol):
    mol = [np.load(m, allow_pickle=True) for m in mol_files]
    n_mol = len(mol)
    species = sorted(list(set([s[0] for m in mol for s in m])))
    
    mol_counts = []
    for name, m in zip(mol_files,mol):
        count_species = {s:0 for s in species}
        for a in m:
            count_species[a[0]] += 1
        mol_counts.append(count_species)

    max_atoms = {s:0 for s in species}
    for m in mol_counts:
        for k, v in m.items():
            max_atoms[k] = max([v, max_atoms[k]])
    max_size = sum(max_atoms.values())
    print(max_atoms, max_size)
    K_global = np.zeros((n_mol, n_mol))
    for m in range(0, len(mol)):
        for n in range(0, len(mol)):
            K_pair = get_covariance(mol[m], mol[n], max_atoms)
            K_global[m][n] = avg_kernel(K_pair)




    print(len(mol_files))





def main():
    X_dir = args.Xdir
    mol_files = [os.path.join(X_dir, f) for f in os.listdir(X_dir) if os.path.isfile(os.path.join(X_dir, f))]
    
    if DEBUG:
        print("Using sub-set = 1000 molecules")
        mol_files = mol_files[:1000]



    return 0




if __name__ == '__main__': main()
