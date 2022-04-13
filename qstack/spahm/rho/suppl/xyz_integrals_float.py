#!/usr/bin/env python3

import sys

def xyz(n,m,k):
# computes the integral of x^2k y^2n z^2m over a sphere
  k,n,m = sorted([k,n,m], reverse=True)
  # k>=n>=m
  if n==0:
    xyz = 2.0 * (1.0 - (2.0*k-1.0)/(2.0*k+1.0))
  else:
    xyz = (2*k-1) * I23(n,m,k)
  return xyz

def I23(n,m,k):
  I23 = 0.0
  for l in range(0, n+m+2):
    I23 = I23 + (-1)**l * trinomial( n+m+1, n+m+1-l, l) / (2.0*l+2.0*k-1.0)
  I23 = I23 / ( (2*n+1) * 2**(2*n+2*m) )
  for l in range(1, n+2):
    I23 = I23 * (2*n+3-2*l) / (2*m-1+2*l)
  return I23

def trinomial(k1,k2,k3):
# (k1+k2+k3)! / (k1! * k2! * k3!)
  k1,k2,k3 = sorted([k1,k2,k3], reverse=True)
  trinom = 1.0
  for k in range(1,k2+1):
    trinom = trinom * (k+k1) / k
  for k in range(1,k3+1):
    trinom = trinom * (k+k1+k2) / k
  return trinom

if __name__ == "__main__":
  k,n,m = map(int, sys.argv[1:4])
  print("%.15f π"%xyz(k,n,m))

