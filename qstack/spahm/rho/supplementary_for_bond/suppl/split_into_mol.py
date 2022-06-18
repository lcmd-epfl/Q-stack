#!/usr/bin/env python3

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split representation by elements')
parser.add_argument('--d', type=str,  dest='dfile',  required=True, help='representation')
parser.add_argument('--n', type=str,  dest='nfile',  required=True, help='list of numbers of atoms')
args = parser.parse_args()

txt = False
try:
  data = list(np.load(args.dfile))
except:
  data = list(np.loadtxt(args.dfile))
  txt  = True
N = np.loadtxt(args.nfile, dtype=int)

for i,n in enumerate(N):
    print(i)
    d = np.array(data[0:n])
    del(data[0:n])
    if txt:
        np.savetxt('mol_%04d.xyz.dat'%i, d)
    else:
        np.save('mol_%04d.xyz.npy'%i, d)
