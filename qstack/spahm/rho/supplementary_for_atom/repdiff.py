import sys
import numpy as np
A = np.load(sys.argv[1]);
B = np.load(sys.argv[2])
for a,b in zip(A,B):
    print(np.linalg.norm(a[1]-b[1]))
