#!/usr/bin/env python3

import numpy as np

data = np.loadtxt('6666')
qs   = np.loadtxt('q.dat', dtype=int)

mydict = {}
for q in sorted(list(set(qs))):
  mydict[q] = []

for q,d in zip(qs,data):
  mydict[q].append(d)

for q in mydict:
  np.save(str(q), np.array(mydict[q]))

