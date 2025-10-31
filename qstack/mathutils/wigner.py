#!/usr/bin/env python3

import sys
import sympy as sp
from .xyz_integrals_sym import xyz as xyzint
from sympy import symbols, Symbol, expand, cancel, expand_trig, Ynm, Ynm_c, Matrix, poly, zeros


# variables
x,y,z    = symbols('x y z')
x1,y1,z1 = symbols('x1 y1 z1')
# coefficients
xx,xy,xz = symbols('xx xy xz')
yx,yy,yz = symbols('yx yy yz')
zx,zy,zz = symbols('zx zy zz')

def real_Y_correct_phase(l, m, theta, phi):
    """Returns real spherical harmonic in Condon-Shortley phase convention.
    
    Note: sympy's Ynm uses a different convention.

    Args:
        l (int): Orbital angular momentum quantum number.
        m (int): Magnetic quantum number.
        theta (sympy.Symbol): Polar angle.
        phi (sympy.Symbol): Azimuthal angle.

    Returns:
        sympy.Expr: Real spherical harmonic expression.
    """
    ym1 = Ynm  (l, -abs(m), theta, phi)
    ym2 = Ynm_c(l, -abs(m), theta, phi)
    if m==0:
        return ym1
    elif m<0:
        return sp.I / sp.sqrt(2) * (ym1 - ym2)
    elif m>0:
        return 1 / sp.sqrt(2) * (ym1 + ym2)

def get_polynom_Y(l, m):
    """Rewrites a real spherical harmonic as a polynomial of x, y, z.

    Args:
        l (int): Orbital angular momentum quantum number.
        m (int): Magnetic quantum number.

    Returns:
        sympy.Expr: Polynomial expression in Cartesian coordinates.
    """
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
    """Wrapper for xyz integrals with caching.

    Args:
        knm (tuple): Tuple of three integers (k, n, m) representing powers.
        integrals_xyz_dict (dict): Cache dictionary for computed integrals.

    Returns:
        float or sympy.Expr: Integral value, or 0 if any power is odd.
    """
    k,n,m = knm
    if k%2 or n%2 or m%2:
        return 0
    else:
        knm = tuple(sorted([k//2, n//2, m//2], reverse=True))
        if knm not in integrals_xyz_dict:
            integrals_xyz_dict[knm] = xyzint(*knm)
        return integrals_xyz_dict[knm]

def product_Y(Y1,Y2):
    """Computes the product of two spherical harmonics.

    Args:
        Y1 (sympy.Expr): First spherical harmonic polynomial.
        Y2 (sympy.Expr): Second spherical harmonic polynomial.

    Returns:
        tuple: A tuple containing:
            - coefficients (sympy.Matrix): Coefficients of the product.
            - monomials (list): List of monomial powers.
    """
    prod = Y1 * Y2
    prod = prod.expand().cancel()
    prod = poly(prod, gens=[x,y,z])
    return Matrix(prod.coeffs()), prod.monoms()


def print_wigner(D):
    """Print Wigner D matrices in formatted output.

    Args:
        D (list): List of Wigner D matrices for each l value.
    """
    for l,d in enumerate(D):
        for m1 in range(-l,l+1):
            for m2 in range(-l,l+1):
                print(f'D[{l}][{m1: d},{m2: d}] = {d[m1,m2]}')
        print()

def compute_wigner(lmax):
    """Compute Wigner D matrices up to a maximum angular momentum.

    Args:
        lmax (int): Maximum angular momentum quantum number.

    Returns:
        list: List of Wigner D matrices (sympy.Matrix) for each l from 0 to lmax.
    """
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

