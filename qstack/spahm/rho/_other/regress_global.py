import os
import sys
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Script to regresse global molecular properties from atomic environnments descriptor (RHO-SPAHM).')
parser.add_argument('--x-dir',  type=str,   dest = 'Xdir',      required = True,            help = 'the directory locating all atomic representation per molecular-file (.npy)')
parser.add_argument('--eta',   type=str,  dest = 'ETA',      required = False,            help = 'optimized eta')
parser.add_argument('--sigma',   type=float,    dest = 'SIGMA',      required = False,            help = 'optimized sigma')
parser.add_argument('--y-file',  type=str,   dest = 'Yfile',      required = False,         help = 'the file containing the target labels (.txt)')
parser.add_argument('--debug',              dest = 'DEBUG',     required = False,   action = 'store_true',   help = 'the directory locating all atomic representation per molecular-file (.npy)')

args = parser.parse_args()


DEBUG = args.DEBUG
SUB=1000



def main():
    from qstack.regression.regression import regression
    
    print(args)
    exit()
    X_dir = args.Xdir
    mol_files = [os.path.join(X_dir, f) for f in os.listdir(X_dir) if os.path.isfile(os.path.join(X_dir, f))]
    print("Found {len(mol_files)} mol. representations")
    y_all = np.loadtxt(args.Yfile)

    eta =  args.ETA
    sigma =args.SIGMA


    if DEBUG:
        print(f"Using sub-set = {SUB} molecules")
        mol_files = mol_files[:SUB]
        y = y_all[:SUB]
    else:
        y = y_all
    mol = [np.load(m, allow_pickle=True) for m in mol_files]
#    K = get_global_K(mol)
    cov_kernel = "G"
    mae_all =  regression(mol, y, sigma=sigma, eta=eta, akernel="global")

    np.savetxt((f"SUB_{SUB}_" if DEBUG else '') + f"LC_{cov_kernel}_{X_dir.split('/')[-1]}.txt", mae_all)
    return 0


if __name__ == '__main__': main()
