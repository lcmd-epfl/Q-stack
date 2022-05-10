#!/usr/bin/env python3

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split representation by elements')
parser.add_argument('--d', type=str,  dest='dfile',  required=True, help='representation')
parser.add_argument('--q', type=str,  dest='qfile',  required=True, help='list of all atoms')
args = parser.parse_args()

data = np.load(args.dfile)
qs   = np.loadtxt(args.qfile, dtype=str)

mydict = {}
for q in sorted(list(set(qs))):
  mydict[q] = []

for q,d in zip(qs,data):
  mydict[q].append(d)

for q in mydict:
  np.save(str(q), np.array(mydict[q]))

