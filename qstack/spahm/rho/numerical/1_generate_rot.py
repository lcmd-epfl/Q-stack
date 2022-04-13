#!/usr/bin/env python3

import sys
import numpy as np

xyz = sys.argv[1]
N   = int(sys.argv[2])

q = np.loadtxt(xyz, usecols=[0], skiprows=2, dtype=str)
r = np.loadtxt(xyz, usecols=[1,2,3], skiprows=2)

fmt = xyz+'.%0'+str(int(np.ceil(np.log10(N))))+'d'

for i in range(N):
  x = i/N * 2.0*np.pi

  R = np.array([[np.cos(x), -np.sin(x), 0.0 ],
                [np.sin(x),  np.cos(x), 0.0 ],
                [0.0      ,  0.0      , 1.0 ]])

  r_rot = (r@R.T)

  with open(fmt%i, 'w') as f:
    print(len(q), end='\n\n', file=f)
    for qi,ri in zip(q, r_rot):
      print(qi, *ri, file=f)

