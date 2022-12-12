#!/usr/bin/env python3

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split representation by elements')
parser.add_argument('--d', type=str,  dest='dfile',  required=True, help='representation')
parser.add_argument('--q', type=str,  dest='qfile',  required=True, help='list of all atoms')
args = parser.parse_args()

txt = False
try:
  data = np.load(args.dfile)
except:
  data = np.loadtxt(args.dfile)
  txt  = True
qs = np.loadtxt(args.qfile, dtype=str)

mydict = {}
for q in sorted(list(set(qs))):
  mydict[q] = []

for q,d in zip(qs,data):
  mydict[q].append(d)

model = os.path.splitext(os.path.basename(args.dfile))[0]
for q in mydict:
  if txt:
    np.savetxt(str(q)+'_'+model+'.dat', np.array(mydict[q]))
  else:
    np.save(str(q)+'_'+model, np.array(mydict[q]))

