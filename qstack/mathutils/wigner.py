#!/usr/bin/env python3

import sys
import sympy as sp
from sympy import symbols, Symbol, simplify, expand, cancel, expand_trig, Ynm, Ynm_c, Matrix, poly, zeros
from .xyz_integrals_sym import xyz as xyzint


# variables
x,y,z    = symbols('x y z')
x1,y1,z1 = symbols('x1 y1 z1')
# coefficients
xx,xy,xz = symbols('xx xy xz')
yx,yy,yz = symbols('yx yy yz')
zx,zy,zz = symbols('zx zy zz')

def real_Y_correct_phase(l, m, theta, phi):
  # returns real spherical harmonic in Condon--Shortley phase convention
  # (sympy's Znm uses some other convention)
  ym1 = Ynm  (l, -abs(m), theta, phi)
  ym2 = Ynm_c(l, -abs(m), theta, phi)
  if m==0:
    return ym1
  elif m<0:
    return sp.I / sp.sqrt(2) * (ym1 - ym2)
  elif m>0:
    return 1 / sp.sqrt(2) * (ym1 + ym2)

def get_polynom_Y(l, m):
  # rewrites a real spherical harmonic as a polynom of x,y,z
  theta = Symbol("theta", real=True)
  phi = Symbol("phi", real=True)
  r = Symbol('r', nonnegative=True)
  expr = real_Y_correct_phase(l,m, theta, phi) * r**l
  expr = expand(expr, func=True)
  expr = expr.rewrite(sp.cos)#.simplify().trigsimp()
  expr = expand_trig(expr)
  expr = cancel(expr)
  expr = expr.subs({r: sp.sqrt(x*x+y*y+z*z), phi: sp.atan2(y,x), theta: sp.atan2(sp.sqrt(x*x+y*y),z)})
  if m!=0:
    expr = cancel(expr).simplify()
  expr = expr.subs({x*x+y*y: 1-z*z,
                   3*x*x+3*y*y : 3-3*z*z })
  return expr

def xyzint_wrapper(knm, integrals_xyz_dict):
  k,n,m = knm
  if k%2 or n%2 or m%2:
    return 0
  else:
    knm = tuple(sorted([k//2, n//2, m//2], reverse=True))
    if not knm in integrals_xyz_dict.keys():
      integrals_xyz_dict[knm] = xyzint(*knm)
    return integrals_xyz_dict[knm]

def product_Y(Y1,Y2):
  # computes the product of two spherical harmonics
  # and returns coefficients and a list of powers
  prod = Y1 * Y2
  prod = prod.expand().cancel()
  prod = poly(prod, gens=[x,y,z])
  return Matrix(prod.coeffs()), prod.monoms()


def print_wigner(D):
  for l,d in enumerate(D):
    for m1 in range(-l,l+1):
      for m2 in range(-l,l+1):
        print('D[%d][% d,% d] = '%(l,m1,m2), d[m1,m2])
    print()

def compute_wigner(lmax):

  Y     = [ [0]*(2*l+1) for l in range(lmax+1)]
  Y_rot = [ [0]*(2*l+1) for l in range(lmax+1)]
  for l in range(lmax+1):
    for m in range(-l,l+1):
      # spherical harmonic
      Y[l][m] = get_polynom_Y(l, m)
      # rotated spherical harmonic
      Y_rot[l][m] = Y[l][m].subs({x: x1, y:y1, z:z1}).subs({x1:xx*x+xy*y+xz*z, y1:yx*x+yy*y+yz*z, z1:zx*x+zy*y+zz*z})


  D = [zeros(2*l+1,2*l+1) for l in range(lmax+1)]
  integrals_xyz_dict = {}
  for l in range(lmax+1):
    for m1 in range(-l,l+1):
      for m2 in range(-l,l+1):
        coefs, pows = product_Y(Y[l][m2], Y_rot[l][m1])
        mono_integrals = [xyzint_wrapper(p,integrals_xyz_dict) for p in pows]
        D[l][m1,m2] = coefs.dot(mono_integrals).factor() .subs({zx**2+zy**2: 1-zz**2, xx**2+xy**2:1-xz**2, yx**2+yy**2:1-yz**2}).simplify()
  return D


if __name__ == "__main__":
  if len(sys.argv)<2:
    lmax = 2
  else:
    lmax = int(sys.argv[1])

  D = compute_wigner(lmax)
  print_wigner(D)

