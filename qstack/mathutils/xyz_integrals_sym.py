#!/usr/bin/env python3

import sys

try:
    import sympy
except ImportError:
    print("""

ERROR: cannot import sympy. Have you installed qstack with the \"wigner\" option?\n\n
(for instance, with `pip install qstack[wigner]` or `pip install qstack[all]`)

""")
    raise

def xyz(n,m,k):
# computes the integral of x^2k y^2n z^2m over a sphere
  k,n,m = sorted([k,n,m], reverse=True)
  # k>=n>=m
  if n==0:
    K = sympy.symbols('K')
    xyz = (2 * (1 - (2*K-1)/(2*K+1))).subs(K,k)
  else:
    xyz = (2*k-1) * I23(n,m,k)
  return xyz * sympy.pi

def I23(n,m,k):
  I23 = 0.0
  K = sympy.symbols('K')
  for l in range(n+m+2):
    I23 = I23 + (-1)**l * trinomial( n+m+1, n+m+1-l, l) / (2*l+2*K-1)
  I23 = I23.subs(K,k)
  I23 = I23 / ( (2*n+1) * 2**(2*n+2*m) )
  for l in range(1, n+2):
    I23 = I23 * (2*n+3-2*l) / (2*m-1+2*l)
  return I23

def trinomial(k1,k2,k3):
# (k1+k2+k3)! / (k1! * k2! * k3!)
  k1,k2,k3 = sorted([k1,k2,k3])
  trinom = sympy.FallingFactorial(k1+k2+k3, k3) / (sympy.factorial(k1)*sympy.factorial(k2))
  return trinom

if __name__ == "__main__":
  k,n,m = map(int, sys.argv[1:4])
  x = xyz(k,n,m)
  print(f"{x:.15f} = {x}")

