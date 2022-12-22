import os
import sys
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Script to regresse global molecular properties from atomic environnments descriptor (RHO-SPAHM).')
parser.add_argument('--x-dir',  type=str,   dest = 'Xdir',      required = True,            help = 'the directory locating all atomic representation per molecular-file (.npy)')
parser.add_argument('--y-file',  type=str,   dest = 'Yfile',      required = False,         help = 'the file containing the target labels (.txt)')
parser.add_argument('--debug',              dest = 'DEBUG',     required = False,   action = 'store_true',   help = 'the directory locating all atomic representation per molecular-file (.npy)')

args = parser.parse_args()


DEBUG = args.DEBUG
SUB=1000


def mol_to_dict(mol, species):
    mol_dict = {s:[]  for s in species}
    for a in mol:
        mol_dict[a[0]].append(a[1])
    for k in mol_dict.keys():
        mol_dict[k] = np.array(mol_dict[k])
    return mol_dict


def get_covariance(mol1, mol2, max_sizes, akernel=None , sigma=None):
    from qstack.regression.kernel import kernel
    species = sorted(max_sizes.keys())
    mol1_dict = mol_to_dict(mol1, species)
    mol2_dict = mol_to_dict(mol2, species)
    max_size = sum(max_sizes.values())
    K_covar = np.zeros((max_size, max_size))
    idx = 0
    for s in species:
        n1 = len(mol1_dict[s])
        n2 = len(mol2_dict[s])
        s_size = max_sizes[s]
        if n1 == 0 or n2 == 0:
            idx += s_size
            continue
        x1 = np.pad(mol1_dict[s], ((0, s_size - n1),(0,0)), 'constant')
        x2 = np.pad(mol2_dict[s], ((0, s_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+s_size, idx:idx+s_size] = kernel(x1, x2, akernel=akernel, sigma=sigma)
        idx += s_size
    return K_covar

def avg_kernel(kernel):
    avg = np.sum(kernel) / len(kernel)**2
    return avg


def get_global_K(X, Y, sigma, akernel='G'):
    n_x = len(X)
    n_y = len(Y)
    species = sorted(list(set([s[0] for m in np.concatenate((X, Y), axis=0) for s in m])))

    mol_counts = []
    for m in np.concatenate((X, Y), axis=0):
        count_species = {s:0 for s in species}
        for a in m:
            count_species[a[0]] += 1
        mol_counts.append(count_species)

    max_atoms = {s:0 for s in species}
    for m in mol_counts:
        for k, v in m.items():
            max_atoms[k] = max([v, max_atoms[k]])
    max_size = sum(max_atoms.values())
    print(max_atoms, max_size, flush=True)
    K_global = np.zeros((n_x, n_y))
    print("Computing global kernel elements:\n[", sep='', end='', flush=True)
    for m in range(0, n_x):
        for n in range(0, n_y):
            K_pair = get_covariance(X[m], Y[n], max_atoms, akernel=akernel, sigma=sigma)
            K_global[m][n] = avg_kernel(K_pair)
        if ((m+1) / len(X) * 100)%10 == 0:
            print(f"##### {(m+1) / len(X) * 100}% #####", sep='', end='', flush=True)
    print("]", flush=True)
    print(f"Final global kernel has size : {K_global.shape}", flush=True)
    return K_global





def main():
    from qstack.regression import hyperparameters

    X_dir = args.Xdir
    mol_files = [os.path.join(X_dir, f) for f in os.listdir(X_dir) if os.path.isfile(os.path.join(X_dir, f))]
    print(f"Found {len(mol_files)} mol. representations", flush=True)
    y_all = np.loadtxt(args.Yfile)

    if DEBUG:
        print(f"Using sub-set = {SUB} molecules", flush=True)
        mol_files = mol_files[:SUB]
        y = y_all[:SUB]
    else:
        y = y_all
    mol = [np.load(m, allow_pickle=True) for m in mol_files]
#    K = get_global_K(mol)
    cov_kernel = "G"
    errors = hyperparameters.hyperparameters(mol, y, akernel='global')
    print(errors)
    np.savetxt((f"SUB_{SUB}_" if DEBUG else '' )+ f"hyperparameters_{cov_kernel}_{X_dir.split('/')[-1]}.txt", errors)





    return 0




if __name__ == '__main__': main()
