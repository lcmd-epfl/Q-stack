import numpy as np
from numpy import sqrt

def c_split(mol, c):
# works for an uncontracted basis only
  cs = []
  i0 = 0
  for at in mol.aoslice_by_atom():
    for b in range(at[0], at[1]):
      l = mol.bas_angular(b)
      msize = 2*l+1
      cs.append([l, c[i0:i0+msize]])
      i0 += msize
  return cs

def rotate_c(D, cs):
  c_new = []
  for l,ci in cs:
    ci_new = D[l] @ ci
    c_new.append(ci_new)
  return np.hstack(c_new)

def new_xy_axis(z):
  # finds the "optimal" axes x' and y' from z'
  z     = z/np.linalg.norm(z)    # don't use /= so a copy of z is created
  i     = np.argmin(abs(z))      # find the axis with the minimal projection of the vector z
  x     = -z[i] * z
  x[i] += 1.0                    # create a vector orthogonal to z with dominant component i
  x    /= np.sqrt(1.0-z[i]*z[i]) # normalize
  y     = np.cross(z,x)
  return np.array([x,y,z])


def Dmatrix(xyz, lmax, order='xyz'):
  # generate Wigner D-matrices D[l][m1,m2] = D_{m1,m2}^l
  # for a rotation encoded as x'=xyz[0], y'=xyz[1], z'=xyz[2]
  # (m1 is rotated so D is transposed)

  xx = xyz[0,0]; xy = xyz[0,1]; xz = xyz[0,2]
  yx = xyz[1,0]; yy = xyz[1,1]; yz = xyz[1,2]
  zx = xyz[2,0]; zy = xyz[2,1]; zz = xyz[2,2]

  SQRT3 = sqrt(3.0)

  D = [np.zeros((2*l+1,2*l+1)) for l in range(lmax+1)]

  D[0][0,0] = 1.0

  if lmax < 1:
      return D

  l=1
  if order=='yzx':  # -1 0 1
    D[1][l+ -1,l+ -1] = yy
    D[1][l+ -1,l+  0] = yz
    D[1][l+ -1,l+  1] = yx
    D[1][l+  0,l+ -1] = zy
    D[1][l+  0,l+  0] = zz
    D[1][l+  0,l+  1] = zx
    D[1][l+  1,l+ -1] = xy
    D[1][l+  1,l+  0] = xz
    D[1][l+  1,l+  1] = xx
  elif order=='xyz': # 1 -1 0
    D[1][ 0, 0] = xx
    D[1][ 0, 1] = xy
    D[1][ 0, 2] = xz
    D[1][ 1, 0] = yx
    D[1][ 1, 1] = yy
    D[1][ 1, 2] = yz
    D[1][ 2, 0] = zx
    D[1][ 2, 1] = zy
    D[1][ 2, 2] = zz

  if lmax < 2:
      return D

  l=2
  D[2][l+ -2,l+ -2] = xx*yy+xy*yx
  D[2][l+ -2,l+ -1] = xy*yz+xz*yy
  D[2][l+ -2,l+  0] = xz*yz * SQRT3
  D[2][l+ -2,l+  1] = xx*yz+xz*yx
  D[2][l+ -2,l+  2] = xx*yx-xy*yy
  D[2][l+ -1,l+ -2] = yx*zy+yy*zx
  D[2][l+ -1,l+ -1] = yy*zz+yz*zy
  D[2][l+ -1,l+  0] = yz*zz * SQRT3
  D[2][l+ -1,l+  1] = yx*zz+yz*zx
  D[2][l+ -1,l+  2] = yx*zx-yy*zy
  D[2][l+  0,l+ -2] = zx*zy * SQRT3
  D[2][l+  0,l+ -1] = zy*zz * SQRT3
  D[2][l+  0,l+  0] = 1.5*zz*zz - 0.5
  D[2][l+  0,l+  1] = zx*zz * SQRT3
  D[2][l+  0,l+  2] = (zx*zx-zy*zy) * 0.5 * SQRT3
  D[2][l+  1,l+ -2] = xx*zy+xy*zx
  D[2][l+  1,l+ -1] = xy*zz+xz*zy
  D[2][l+  1,l+  0] = xz*zz * SQRT3
  D[2][l+  1,l+  1] = xx*zz+xz*zx
  D[2][l+  1,l+  2] = xx*zx-xy*zy
  D[2][l+  2,l+ -2] = xx*xy-yx*yy
  D[2][l+  2,l+ -1] = xy*xz-yy*yz
  D[2][l+  2,l+  0] = (xz*xz-yz*yz) * 0.5 * SQRT3
  D[2][l+  2,l+  1] = xx*xz-yx*yz
  D[2][l+  2,l+  2] = (xx*xx-xy*xy+yy*yy-yx*yx) * 0.5

  if lmax < 3:
      return D

  l=3
  D[3][l+ -3,l+ -3] =  3*xx**2*yy/4 + 3*xx*xy*yx/2 - 3*xy**2*yy/4 - 3*yx**2*yy/4 + yy**3/4
  D[3][l+ -3,l+ -2] =  sqrt(6)*(xx*xy*yz + xx*xz*yy + xy*xz*yx - yx*yy*yz)/2
  D[3][l+ -3,l+ -1] =  sqrt(15)*(-xx**2*yy - 2*xx*xy*yx - 3*xy**2*yy + 8*xy*xz*yz + 4*xz**2*yy + yx**2*yy + yy**3 - 4*yy*yz**2)/20
  D[3][l+ -3,l+  0] =  sqrt(10)*(-3*xx**2*yz - 6*xx*xz*yx - 3*xy**2*yz - 6*xy*xz*yy + 6*xz**2*yz + 3*yx**2*yz + 3*yy**2*yz - 2*yz**3)/20
  D[3][l+ -3,l+  1] =  sqrt(15)*(-3*xx**2*yx - 2*xx*xy*yy + 8*xx*xz*yz - xy**2*yx + 4*xz**2*yx + yx**3 + yx*yy**2 - 4*yx*yz**2)/20
  D[3][l+ -3,l+  2] =  sqrt(6)*(xx**2*yz + 2*xx*xz*yx - xy**2*yz - 2*xy*xz*yy - yx**2*yz + yy**2*yz)/4
  D[3][l+ -3,l+  3] =  3*xx**2*yx/4 - 3*xx*xy*yy/2 - 3*xy**2*yx/4 - yx**3/4 + 3*yx*yy**2/4
  D[3][l+ -2,l+ -3] =  sqrt(6)*(xx*yx*zy + xx*yy*zx + xy*yx*zx - xy*yy*zy)/2
  D[3][l+ -2,l+ -2] =  xx*yy*zz + xx*yz*zy + xy*yx*zz + xy*yz*zx + xz*yx*zy + xz*yy*zx
  D[3][l+ -2,l+ -1] =  sqrt(10)*(-xx*yx*zy - xx*yy*zx - xy*yx*zx - 3*xy*yy*zy + 4*xy*yz*zz + 4*xz*yy*zz + 4*xz*yz*zy)/10
  D[3][l+ -2,l+  0] =  sqrt(15)*(-xx*yx*zz - xx*yz*zx - xy*yy*zz - xy*yz*zy - xz*yx*zx - xz*yy*zy + 2*xz*yz*zz)/5
  D[3][l+ -2,l+  1] =  sqrt(10)*(-3*xx*yx*zx - xx*yy*zy + 4*xx*yz*zz - xy*yx*zy - xy*yy*zx + 4*xz*yx*zz + 4*xz*yz*zx)/10
  D[3][l+ -2,l+  2] =  xx*yx*zz + xx*yz*zx - xy*yy*zz - xy*yz*zy + xz*yx*zx - xz*yy*zy
  D[3][l+ -2,l+  3] =  sqrt(6)*(xx*yx*zx - xx*yy*zy - xy*yx*zy - xy*yy*zx)/2
  D[3][l+ -1,l+ -3] =  sqrt(15)*(2*yx*zx*zy + yy*zx**2 - yy*zy**2)/4
  D[3][l+ -1,l+ -2] =  sqrt(10)*(yx*zy*zz + yy*zx*zz + yz*zx*zy)/2
  D[3][l+ -1,l+ -1] =  -yx*zx*zy/2 - yy*zx**2/4 - 3*yy*zy**2/4 + yy*zz**2 + 2*yz*zy*zz
  D[3][l+ -1,l+  0] =  sqrt(6)*(-2*yx*zx*zz - 2*yy*zy*zz - yz*zx**2 - yz*zy**2 + 2*yz*zz**2)/4
  D[3][l+ -1,l+  1] =  -3*yx*zx**2/4 - yx*zy**2/4 + yx*zz**2 - yy*zx*zy/2 + 2*yz*zx*zz
  D[3][l+ -1,l+  2] =  sqrt(10)*(2*yx*zx*zz - 2*yy*zy*zz + yz*zx**2 - yz*zy**2)/4
  D[3][l+ -1,l+  3] =  sqrt(15)*(yx*zx**2 - yx*zy**2 - 2*yy*zx*zy)/4
  D[3][l+  0,l+ -3] =  sqrt(10)*zy*(3*zx**2 - zy**2)/4
  D[3][l+  0,l+ -2] =  sqrt(15)*zx*zy*zz
  D[3][l+  0,l+ -1] =  sqrt(6)*zy*(5*zz**2 - 1)/4
  D[3][l+  0,l+  0] =  zz*(-3*zx**2 - 3*zy**2 + 2*zz**2)/2
  D[3][l+  0,l+  1] =  sqrt(6)*zx*(5*zz**2 - 1)/4
  D[3][l+  0,l+  2] =  sqrt(15)*zz*(zx - zy)*(zx + zy)/2
  D[3][l+  0,l+  3] =  sqrt(10)*zx*(zx**2 - 3*zy**2)/4
  D[3][l+  1,l+ -3] =  sqrt(15)*(2*xx*zx*zy + xy*zx**2 - xy*zy**2)/4
  D[3][l+  1,l+ -2] =  sqrt(10)*(xx*zy*zz + xy*zx*zz + xz*zx*zy)/2
  D[3][l+  1,l+ -1] =  -xx*zx*zy/2 - xy*zx**2/4 - 3*xy*zy**2/4 + xy*zz**2 + 2*xz*zy*zz
  D[3][l+  1,l+  0] =  sqrt(6)*(-2*xx*zx*zz - 2*xy*zy*zz - xz*zx**2 - xz*zy**2 + 2*xz*zz**2)/4
  D[3][l+  1,l+  1] =  -3*xx*zx**2/4 - xx*zy**2/4 + xx*zz**2 - xy*zx*zy/2 + 2*xz*zx*zz
  D[3][l+  1,l+  2] =  sqrt(10)*(2*xx*zx*zz - 2*xy*zy*zz + xz*zx**2 - xz*zy**2)/4
  D[3][l+  1,l+  3] =  sqrt(15)*(xx*zx**2 - xx*zy**2 - 2*xy*zx*zy)/4
  D[3][l+  2,l+ -3] =  sqrt(6)*(xx**2*zy + 2*xx*xy*zx - xy**2*zy - yx**2*zy - 2*yx*yy*zx + yy**2*zy)/4
  D[3][l+  2,l+ -2] =  xx*xy*zz + xx*xz*zy + xy*xz*zx - yx*yy*zz - yx*yz*zy - yy*yz*zx
  D[3][l+  2,l+ -1] =  sqrt(10)*(-xx**2*zy - 2*xx*xy*zx - 3*xy**2*zy + 8*xy*xz*zz + 4*xz**2*zy + yx**2*zy + 2*yx*yy*zx + 3*yy**2*zy - 8*yy*yz*zz - 4*yz**2*zy)/20
  D[3][l+  2,l+  0] =  sqrt(15)*(-xx**2*zz - 2*xx*xz*zx - xy**2*zz - 2*xy*xz*zy + 2*xz**2*zz + yx**2*zz + 2*yx*yz*zx + yy**2*zz + 2*yy*yz*zy - 2*yz**2*zz)/10
  D[3][l+  2,l+  1] =  sqrt(10)*(-3*xx**2*zx - 2*xx*xy*zy + 8*xx*xz*zz - xy**2*zx + 4*xz**2*zx + 3*yx**2*zx + 2*yx*yy*zy - 8*yx*yz*zz + yy**2*zx - 4*yz**2*zx)/20
  D[3][l+  2,l+  2] =  xx**2*zz/2 + xx*xz*zx - xy**2*zz/2 - xy*xz*zy - yx**2*zz/2 - yx*yz*zx + yy**2*zz/2 + yy*yz*zy
  D[3][l+  2,l+  3] =  sqrt(6)*(xx**2*zx - 2*xx*xy*zy - xy**2*zx - yx**2*zx + 2*yx*yy*zy + yy**2*zx)/4
  D[3][l+  3,l+ -3] =  3*xx**2*xy/4 - 3*xx*yx*yy/2 - xy**3/4 - 3*xy*yx**2/4 + 3*xy*yy**2/4
  D[3][l+  3,l+ -2] =  sqrt(6)*(xx*xy*xz - xx*yy*yz - xy*yx*yz - xz*yx*yy)/2
  D[3][l+  3,l+ -1] =  sqrt(15)*(-xx**2*xy + 2*xx*yx*yy - xy**3 + 4*xy*xz**2 + xy*yx**2 + 3*xy*yy**2 - 4*xy*yz**2 - 8*xz*yy*yz)/20
  D[3][l+  3,l+  0] =  sqrt(10)*(-3*xx**2*xz + 6*xx*yx*yz - 3*xy**2*xz + 6*xy*yy*yz + 2*xz**3 + 3*xz*yx**2 + 3*xz*yy**2 - 6*xz*yz**2)/20
  D[3][l+  3,l+  1] =  sqrt(15)*(-xx**3 - xx*xy**2 + 4*xx*xz**2 + 3*xx*yx**2 + xx*yy**2 - 4*xx*yz**2 + 2*xy*yx*yy - 8*xz*yx*yz)/20
  D[3][l+  3,l+  2] =  sqrt(6)*(xx**2*xz - 2*xx*yx*yz - xy**2*xz + 2*xy*yy*yz - xz*yx**2 + xz*yy**2)/4
  D[3][l+  3,l+  3] =  xx**3/4 - 3*xx*xy**2/4 - 3*xx*yx**2/4 + 3*xx*yy**2/4 + 3*xy*yx*yy/2

  if lmax < 4:
      return D

  l=4
  D[4][l+ -4,l+-4] =  xx**3*yy/2 + 3*xx**2*xy*yx/2 - 3*xx*xy**2*yy/2 - 3*xx*yx**2*yy/2 + xx*yy**3/2 - xy**3*yx/2 - xy*yx**3/2 + 3*xy*yx*yy**2/2
  D[4][l+ -4,l+-3] =  sqrt(2)*(3*xx**2*xy*yz + 3*xx**2*xz*yy + 6*xx*xy*xz*yx - 6*xx*yx*yy*yz - xy**3*yz - 3*xy**2*xz*yy - 3*xy*yx**2*yz + 3*xy*yy**2*yz - 3*xz*yx**2*yy + xz*yy**3)/4
  D[4][l+ -4,l+-2] =  sqrt(7)*(-xx**3*yy - 3*xx**2*xy*yx - 3*xx*xy**2*yy + 12*xx*xy*xz*yz + 6*xx*xz**2*yy + 3*xx*yx**2*yy + xx*yy**3 - 6*xx*yy*yz**2 - xy**3*yx + 6*xy*xz**2*yx + xy*yx**3 + 3*xy*yx*yy**2 - 6*xy*yx*yz**2 - 12*xz*yx*yy*yz)/14
  D[4][l+ -4,l+-1] =  sqrt(14)*(-3*xx**2*xy*yz - 3*xx**2*xz*yy - 6*xx*xy*xz*yx + 6*xx*yx*yy*yz - 3*xy**3*yz - 9*xy**2*xz*yy + 12*xy*xz**2*yz + 3*xy*yx**2*yz + 9*xy*yy**2*yz - 4*xy*yz**3 + 4*xz**3*yy + 3*xz*yx**2*yy + 3*xz*yy**3 - 12*xz*yy*yz**2)/28
  D[4][l+ -4,l+ 0] =  sqrt(35)*(3*xx**3*yx + 3*xx**2*xy*yy - 12*xx**2*xz*yz + 3*xx*xy**2*yx - 12*xx*xz**2*yx - 3*xx*yx**3 - 3*xx*yx*yy**2 + 12*xx*yx*yz**2 + 3*xy**3*yy - 12*xy**2*xz*yz - 12*xy*xz**2*yy - 3*xy*yx**2*yy - 3*xy*yy**3 + 12*xy*yy*yz**2 + 8*xz**3*yz + 12*xz*yx**2*yz + 12*xz*yy**2*yz - 8*xz*yz**3)/70
  D[4][l+ -4,l+ 1] =  sqrt(14)*(-3*xx**3*yz - 9*xx**2*xz*yx - 3*xx*xy**2*yz - 6*xx*xy*xz*yy + 12*xx*xz**2*yz + 9*xx*yx**2*yz + 3*xx*yy**2*yz - 4*xx*yz**3 - 3*xy**2*xz*yx + 6*xy*yx*yy*yz + 4*xz**3*yx + 3*xz*yx**3 + 3*xz*yx*yy**2 - 12*xz*yx*yz**2)/28
  D[4][l+ -4,l+ 2] =  sqrt(7)*(-xx**3*yx + 3*xx**2*xz*yz + 3*xx*xz**2*yx + xx*yx**3 - 3*xx*yx*yz**2 + xy**3*yy - 3*xy**2*xz*yz - 3*xy*xz**2*yy - xy*yy**3 + 3*xy*yy*yz**2 - 3*xz*yx**2*yz + 3*xz*yy**2*yz)/7
  D[4][l+ -4,l+ 3] =  sqrt(2)*(xx**3*yz + 3*xx**2*xz*yx - 3*xx*xy**2*yz - 6*xx*xy*xz*yy - 3*xx*yx**2*yz + 3*xx*yy**2*yz - 3*xy**2*xz*yx + 6*xy*yx*yy*yz - xz*yx**3 + 3*xz*yx*yy**2)/4
  D[4][l+ -4,l+ 4] =  xx**3*yx/2 - 3*xx**2*xy*yy/2 - 3*xx*xy**2*yx/2 - xx*yx**3/2 + 3*xx*yx*yy**2/2 + xy**3*yy/2 + 3*xy*yx**2*yy/2 - xy*yy**3/2
  D[4][l+ -3,l+-4] =  sqrt(2)*(3*xx**2*yx*zy + 3*xx**2*yy*zx + 6*xx*xy*yx*zx - 6*xx*xy*yy*zy - 3*xy**2*yx*zy - 3*xy**2*yy*zx - yx**3*zy - 3*yx**2*yy*zx + 3*yx*yy**2*zy + yy**3*zx)/4
  D[4][l+ -3,l+-3] =  3*xx**2*yy*zz/4 + 3*xx**2*yz*zy/4 + 3*xx*xy*yx*zz/2 + 3*xx*xy*yz*zx/2 + 3*xx*xz*yx*zy/2 + 3*xx*xz*yy*zx/2 - 3*xy**2*yy*zz/4 - 3*xy**2*yz*zy/4 + 3*xy*xz*yx*zx/2 - 3*xy*xz*yy*zy/2 - 3*yx**2*yy*zz/4 - 3*yx**2*yz*zy/4 - 3*yx*yy*yz*zx/2 + yy**3*zz/4 + 3*yy**2*yz*zy/4
  D[4][l+ -3,l+-2] =  sqrt(14)*(-3*xx**2*yx*zy - 3*xx**2*yy*zx - 6*xx*xy*yx*zx - 6*xx*xy*yy*zy + 12*xx*xy*yz*zz + 12*xx*xz*yy*zz + 12*xx*xz*yz*zy - 3*xy**2*yx*zy - 3*xy**2*yy*zx + 12*xy*xz*yx*zz + 12*xy*xz*yz*zx + 6*xz**2*yx*zy + 6*xz**2*yy*zx + yx**3*zy + 3*yx**2*yy*zx + 3*yx*yy**2*zy - 12*yx*yy*yz*zz - 6*yx*yz**2*zy + yy**3*zx - 6*yy*yz**2*zx)/28
  D[4][l+ -3,l+-1] =  sqrt(7)*(-3*xx**2*yy*zz - 3*xx**2*yz*zy - 6*xx*xy*yx*zz - 6*xx*xy*yz*zx - 6*xx*xz*yx*zy - 6*xx*xz*yy*zx - 9*xy**2*yy*zz - 9*xy**2*yz*zy - 6*xy*xz*yx*zx - 18*xy*xz*yy*zy + 24*xy*xz*yz*zz + 12*xz**2*yy*zz + 12*xz**2*yz*zy + 3*yx**2*yy*zz + 3*yx**2*yz*zy + 6*yx*yy*yz*zx + 3*yy**3*zz + 9*yy**2*yz*zy - 12*yy*yz**2*zz - 4*yz**3*zy)/28
  D[4][l+ -3,l+ 0] =  sqrt(70)*(9*xx**2*yx*zx + 3*xx**2*yy*zy - 12*xx**2*yz*zz + 6*xx*xy*yx*zy + 6*xx*xy*yy*zx - 24*xx*xz*yx*zz - 24*xx*xz*yz*zx + 3*xy**2*yx*zx + 9*xy**2*yy*zy - 12*xy**2*yz*zz - 24*xy*xz*yy*zz - 24*xy*xz*yz*zy - 12*xz**2*yx*zx - 12*xz**2*yy*zy + 24*xz**2*yz*zz - 3*yx**3*zx - 3*yx**2*yy*zy + 12*yx**2*yz*zz - 3*yx*yy**2*zx + 12*yx*yz**2*zx - 3*yy**3*zy + 12*yy**2*yz*zz + 12*yy*yz**2*zy - 8*yz**3*zz)/140
  D[4][l+ -3,l+ 1] =  sqrt(7)*(-9*xx**2*yx*zz - 9*xx**2*yz*zx - 6*xx*xy*yy*zz - 6*xx*xy*yz*zy - 18*xx*xz*yx*zx - 6*xx*xz*yy*zy + 24*xx*xz*yz*zz - 3*xy**2*yx*zz - 3*xy**2*yz*zx - 6*xy*xz*yx*zy - 6*xy*xz*yy*zx + 12*xz**2*yx*zz + 12*xz**2*yz*zx + 3*yx**3*zz + 9*yx**2*yz*zx + 3*yx*yy**2*zz + 6*yx*yy*yz*zy - 12*yx*yz**2*zz + 3*yy**2*yz*zx - 4*yz**3*zx)/28
  D[4][l+ -3,l+ 2] =  sqrt(14)*(-3*xx**2*yx*zx + 3*xx**2*yz*zz + 6*xx*xz*yx*zz + 6*xx*xz*yz*zx + 3*xy**2*yy*zy - 3*xy**2*yz*zz - 6*xy*xz*yy*zz - 6*xy*xz*yz*zy + 3*xz**2*yx*zx - 3*xz**2*yy*zy + yx**3*zx - 3*yx**2*yz*zz - 3*yx*yz**2*zx - yy**3*zy + 3*yy**2*yz*zz + 3*yy*yz**2*zy)/14
  D[4][l+ -3,l+ 3] =  3*xx**2*yx*zz/4 + 3*xx**2*yz*zx/4 - 3*xx*xy*yy*zz/2 - 3*xx*xy*yz*zy/2 + 3*xx*xz*yx*zx/2 - 3*xx*xz*yy*zy/2 - 3*xy**2*yx*zz/4 - 3*xy**2*yz*zx/4 - 3*xy*xz*yx*zy/2 - 3*xy*xz*yy*zx/2 - yx**3*zz/4 - 3*yx**2*yz*zx/4 + 3*yx*yy**2*zz/4 + 3*yx*yy*yz*zy/2 + 3*yy**2*yz*zx/4
  D[4][l+ -3,l+ 4] =  sqrt(2)*(3*xx**2*yx*zx - 3*xx**2*yy*zy - 6*xx*xy*yx*zy - 6*xx*xy*yy*zx - 3*xy**2*yx*zx + 3*xy**2*yy*zy - yx**3*zx + 3*yx**2*yy*zy + 3*yx*yy**2*zx - yy**3*zy)/4
  D[4][l+ -2,l+-4] =  sqrt(7)*(2*xx*yx*zx*zy + xx*yy*zx**2 - xx*yy*zy**2 + xy*yx*zx**2 - xy*yx*zy**2 - 2*xy*yy*zx*zy)/2
  D[4][l+ -2,l+-3] =  sqrt(14)*(2*xx*yx*zy*zz + 2*xx*yy*zx*zz + 2*xx*yz*zx*zy + 2*xy*yx*zx*zz - 2*xy*yy*zy*zz + xy*yz*zx**2 - xy*yz*zy**2 + 2*xz*yx*zx*zy + xz*yy*zx**2 - xz*yy*zy**2)/4
  D[4][l+ -2,l+-2] =  -xx*yx*zx*zy - xx*yy*zx**2/2 - xx*yy*zy**2/2 + xx*yy*zz**2 + 2*xx*yz*zy*zz - xy*yx*zx**2/2 - xy*yx*zy**2/2 + xy*yx*zz**2 - xy*yy*zx*zy + 2*xy*yz*zx*zz + 2*xz*yx*zy*zz + 2*xz*yy*zx*zz + 2*xz*yz*zx*zy
  D[4][l+ -2,l+-1] =  sqrt(2)*(-2*xx*yx*zy*zz - 2*xx*yy*zx*zz - 2*xx*yz*zx*zy - 2*xy*yx*zx*zz - 6*xy*yy*zy*zz - xy*yz*zx**2 - 3*xy*yz*zy**2 + 4*xy*yz*zz**2 - 2*xz*yx*zx*zy - xz*yy*zx**2 - 3*xz*yy*zy**2 + 4*xz*yy*zz**2 + 8*xz*yz*zy*zz)/4
  D[4][l+ -2,l+ 0] =  sqrt(5)*(3*xx*yx*zx**2 + xx*yx*zy**2 - 4*xx*yx*zz**2 + 2*xx*yy*zx*zy - 8*xx*yz*zx*zz + 2*xy*yx*zx*zy + xy*yy*zx**2 + 3*xy*yy*zy**2 - 4*xy*yy*zz**2 - 8*xy*yz*zy*zz - 8*xz*yx*zx*zz - 8*xz*yy*zy*zz - 4*xz*yz*zx**2 - 4*xz*yz*zy**2 + 8*xz*yz*zz**2)/10
  D[4][l+ -2,l+ 1] =  sqrt(2)*(-6*xx*yx*zx*zz - 2*xx*yy*zy*zz - 3*xx*yz*zx**2 - xx*yz*zy**2 + 4*xx*yz*zz**2 - 2*xy*yx*zy*zz - 2*xy*yy*zx*zz - 2*xy*yz*zx*zy - 3*xz*yx*zx**2 - xz*yx*zy**2 + 4*xz*yx*zz**2 - 2*xz*yy*zx*zy + 8*xz*yz*zx*zz)/4
  D[4][l+ -2,l+ 2] =  -xx*yx*zx**2 + xx*yx*zz**2 + 2*xx*yz*zx*zz + xy*yy*zy**2 - xy*yy*zz**2 - 2*xy*yz*zy*zz + 2*xz*yx*zx*zz - 2*xz*yy*zy*zz + xz*yz*zx**2 - xz*yz*zy**2
  D[4][l+ -2,l+ 3] =  sqrt(14)*(2*xx*yx*zx*zz - 2*xx*yy*zy*zz + xx*yz*zx**2 - xx*yz*zy**2 - 2*xy*yx*zy*zz - 2*xy*yy*zx*zz - 2*xy*yz*zx*zy + xz*yx*zx**2 - xz*yx*zy**2 - 2*xz*yy*zx*zy)/4
  D[4][l+ -2,l+ 4] =  sqrt(7)*(xx*yx*zx**2 - xx*yx*zy**2 - 2*xx*yy*zx*zy - 2*xy*yx*zx*zy - xy*yy*zx**2 + xy*yy*zy**2)/2
  D[4][l+ -1,l+-4] =  sqrt(14)*(3*yx*zx**2*zy - yx*zy**3 + yy*zx**3 - 3*yy*zx*zy**2)/4
  D[4][l+ -1,l+-3] =  sqrt(7)*(6*yx*zx*zy*zz + 3*yy*zx**2*zz - 3*yy*zy**2*zz + 3*yz*zx**2*zy - yz*zy**3)/4
  D[4][l+ -1,l+-2] =  sqrt(2)*(-3*yx*zx**2*zy - yx*zy**3 + 6*yx*zy*zz**2 - yy*zx**3 - 3*yy*zx*zy**2 + 6*yy*zx*zz**2 + 12*yz*zx*zy*zz)/4
  D[4][l+ -1,l+-1] =  -3*yx*zx*zy*zz/2 - 3*yy*zx**2*zz/4 - 9*yy*zy**2*zz/4 + yy*zz**3 - 3*yz*zx**2*zy/4 - 3*yz*zy**3/4 + 3*yz*zy*zz**2
  D[4][l+ -1,l+ 0] =  sqrt(10)*(3*yx*zx**3 + 3*yx*zx*zy**2 - 12*yx*zx*zz**2 + 3*yy*zx**2*zy + 3*yy*zy**3 - 12*yy*zy*zz**2 - 12*yz*zx**2*zz - 12*yz*zy**2*zz + 8*yz*zz**3)/20
  D[4][l+ -1,l+ 1] =  -9*yx*zx**2*zz/4 - 3*yx*zy**2*zz/4 + yx*zz**3 - 3*yy*zx*zy*zz/2 - 3*yz*zx**3/4 - 3*yz*zx*zy**2/4 + 3*yz*zx*zz**2
  D[4][l+ -1,l+ 2] =  sqrt(2)*(-yx*zx**3 + 3*yx*zx*zz**2 + yy*zy**3 - 3*yy*zy*zz**2 + 3*yz*zx**2*zz - 3*yz*zy**2*zz)/2
  D[4][l+ -1,l+ 3] =  sqrt(7)*(3*yx*zx**2*zz - 3*yx*zy**2*zz - 6*yy*zx*zy*zz + yz*zx**3 - 3*yz*zx*zy**2)/4
  D[4][l+ -1,l+ 4] =  sqrt(14)*(yx*zx**3 - 3*yx*zx*zy**2 - 3*yy*zx**2*zy + yy*zy**3)/4
  D[4][l+  0,l+-4] =  sqrt(35)*zx*zy*(zx - zy)*(zx + zy)/2
  D[4][l+  0,l+-3] =  sqrt(70)*zy*zz*(3*zx**2 - zy**2)/4
  D[4][l+  0,l+-2] =  sqrt(5)*zx*zy*(7*zz**2 - 1)/2
  D[4][l+  0,l+-1] =  sqrt(10)*zy*zz*(-3*zx**2 - 3*zy**2 + 4*zz**2)/4
  D[4][l+  0,l+ 0] =  3*zx**4/8 + 3*zx**2*zy**2/4 - 3*zx**2*zz**2 + 3*zy**4/8 - 3*zy**2*zz**2 + zz**4
  D[4][l+  0,l+ 1] =  sqrt(10)*zx*zz*(-3*zx**2 - 3*zy**2 + 4*zz**2)/4
  D[4][l+  0,l+ 2] =  sqrt(5)*(zx - zy)*(zx + zy)*(7*zz**2 - 1)/4
  D[4][l+  0,l+ 3] =  sqrt(70)*zx*zz*(zx**2 - 3*zy**2)/4
  D[4][l+  0,l+ 4] =  sqrt(35)*(zx**4 - 6*zx**2*zy**2 + zy**4)/8
  D[4][l+  1,l+-4] =  sqrt(14)*(3*xx*zx**2*zy - xx*zy**3 + xy*zx**3 - 3*xy*zx*zy**2)/4
  D[4][l+  1,l+-3] =  sqrt(7)*(6*xx*zx*zy*zz + 3*xy*zx**2*zz - 3*xy*zy**2*zz + 3*xz*zx**2*zy - xz*zy**3)/4
  D[4][l+  1,l+-2] =  sqrt(2)*(-3*xx*zx**2*zy - xx*zy**3 + 6*xx*zy*zz**2 - xy*zx**3 - 3*xy*zx*zy**2 + 6*xy*zx*zz**2 + 12*xz*zx*zy*zz)/4
  D[4][l+  1,l+-1] =  -3*xx*zx*zy*zz/2 - 3*xy*zx**2*zz/4 - 9*xy*zy**2*zz/4 + xy*zz**3 - 3*xz*zx**2*zy/4 - 3*xz*zy**3/4 + 3*xz*zy*zz**2
  D[4][l+  1,l+ 0] =  sqrt(10)*(3*xx*zx**3 + 3*xx*zx*zy**2 - 12*xx*zx*zz**2 + 3*xy*zx**2*zy + 3*xy*zy**3 - 12*xy*zy*zz**2 - 12*xz*zx**2*zz - 12*xz*zy**2*zz + 8*xz*zz**3)/20
  D[4][l+  1,l+ 1] =  -9*xx*zx**2*zz/4 - 3*xx*zy**2*zz/4 + xx*zz**3 - 3*xy*zx*zy*zz/2 - 3*xz*zx**3/4 - 3*xz*zx*zy**2/4 + 3*xz*zx*zz**2
  D[4][l+  1,l+ 2] =  sqrt(2)*(-xx*zx**3 + 3*xx*zx*zz**2 + xy*zy**3 - 3*xy*zy*zz**2 + 3*xz*zx**2*zz - 3*xz*zy**2*zz)/2
  D[4][l+  1,l+ 3] =  sqrt(7)*(3*xx*zx**2*zz - 3*xx*zy**2*zz - 6*xy*zx*zy*zz + xz*zx**3 - 3*xz*zx*zy**2)/4
  D[4][l+  1,l+ 4] =  sqrt(14)*(xx*zx**3 - 3*xx*zx*zy**2 - 3*xy*zx**2*zy + xy*zy**3)/4
  D[4][l+  2,l+-4] =  sqrt(7)*(-xx**3*xy + 3*xx**2*zx*zy + xx*xy**3 + 3*xx*xy*zx**2 - 3*xx*xy*zy**2 - 3*xy**2*zx*zy + yx**3*yy - 3*yx**2*zx*zy - yx*yy**3 - 3*yx*yy*zx**2 + 3*yx*yy*zy**2 + 3*yy**2*zx*zy)/7
  D[4][l+  2,l+-3] =  sqrt(14)*(-3*xx**2*xy*xz + 3*xx**2*zy*zz + 6*xx*xy*zx*zz + 6*xx*xz*zx*zy + xy**3*xz - 3*xy**2*zy*zz + 3*xy*xz*zx**2 - 3*xy*xz*zy**2 + 3*yx**2*yy*yz - 3*yx**2*zy*zz - 6*yx*yy*zx*zz - 6*yx*yz*zx*zy - yy**3*yz + 3*yy**2*zy*zz - 3*yy*yz*zx**2 + 3*yy*yz*zy**2)/14
  D[4][l+  2,l+-2] =  xx**3*xy/7 - 3*xx**2*zx*zy/7 + xx*xy**3/7 - 6*xx*xy*xz**2/7 - 3*xx*xy*zx**2/7 - 3*xx*xy*zy**2/7 + 6*xx*xy*zz**2/7 + 12*xx*xz*zy*zz/7 - 3*xy**2*zx*zy/7 + 12*xy*xz*zx*zz/7 + 6*xz**2*zx*zy/7 - yx**3*yy/7 + 3*yx**2*zx*zy/7 - yx*yy**3/7 + 6*yx*yy*yz**2/7 + 3*yx*yy*zx**2/7 + 3*yx*yy*zy**2/7 - 6*yx*yy*zz**2/7 - 12*yx*yz*zy*zz/7 + 3*yy**2*zx*zy/7 - 12*yy*yz*zx*zz/7 - 6*yz**2*zx*zy/7
  D[4][l+  2,l+-1] =  sqrt(2)*(3*xx**2*xy*xz - 3*xx**2*zy*zz - 6*xx*xy*zx*zz - 6*xx*xz*zx*zy + 3*xy**3*xz - 9*xy**2*zy*zz - 4*xy*xz**3 - 3*xy*xz*zx**2 - 9*xy*xz*zy**2 + 12*xy*xz*zz**2 + 12*xz**2*zy*zz - 3*yx**2*yy*yz + 3*yx**2*zy*zz + 6*yx*yy*zx*zz + 6*yx*yz*zx*zy - 3*yy**3*yz + 9*yy**2*zy*zz + 4*yy*yz**3 + 3*yy*yz*zx**2 + 9*yy*yz*zy**2 - 12*yy*yz*zz**2 - 12*yz**2*zy*zz)/14
  D[4][l+  2,l+ 0] =  sqrt(5)*(-3*xx**4 - 6*xx**2*xy**2 + 24*xx**2*xz**2 + 18*xx**2*zx**2 + 6*xx**2*zy**2 - 24*xx**2*zz**2 + 24*xx*xy*zx*zy - 96*xx*xz*zx*zz - 3*xy**4 + 24*xy**2*xz**2 + 6*xy**2*zx**2 + 18*xy**2*zy**2 - 24*xy**2*zz**2 - 96*xy*xz*zy*zz - 8*xz**4 - 24*xz**2*zx**2 - 24*xz**2*zy**2 + 48*xz**2*zz**2 + 3*yx**4 + 6*yx**2*yy**2 - 24*yx**2*yz**2 - 18*yx**2*zx**2 - 6*yx**2*zy**2 + 24*yx**2*zz**2 - 24*yx*yy*zx*zy + 96*yx*yz*zx*zz + 3*yy**4 - 24*yy**2*yz**2 - 6*yy**2*zx**2 - 18*yy**2*zy**2 + 24*yy**2*zz**2 + 96*yy*yz*zy*zz + 8*yz**4 + 24*yz**2*zx**2 + 24*yz**2*zy**2 - 48*yz**2*zz**2)/140
  D[4][l+  2,l+ 1] =  sqrt(2)*(3*xx**3*xz - 9*xx**2*zx*zz + 3*xx*xy**2*xz - 6*xx*xy*zy*zz - 4*xx*xz**3 - 9*xx*xz*zx**2 - 3*xx*xz*zy**2 + 12*xx*xz*zz**2 - 3*xy**2*zx*zz - 6*xy*xz*zx*zy + 12*xz**2*zx*zz - 3*yx**3*yz + 9*yx**2*zx*zz - 3*yx*yy**2*yz + 6*yx*yy*zy*zz + 4*yx*yz**3 + 9*yx*yz*zx**2 + 3*yx*yz*zy**2 - 12*yx*yz*zz**2 + 3*yy**2*zx*zz + 6*yy*yz*zx*zy - 12*yz**2*zx*zz)/14
  D[4][l+  2,l+ 2] =  xx**4/14 - 3*xx**2*xz**2/7 - 3*xx**2*zx**2/7 + 3*xx**2*zz**2/7 + 12*xx*xz*zx*zz/7 - xy**4/14 + 3*xy**2*xz**2/7 + 3*xy**2*zy**2/7 - 3*xy**2*zz**2/7 - 12*xy*xz*zy*zz/7 + 3*xz**2*zx**2/7 - 3*xz**2*zy**2/7 - yx**4/14 + 3*yx**2*yz**2/7 + 3*yx**2*zx**2/7 - 3*yx**2*zz**2/7 - 12*yx*yz*zx*zz/7 + yy**4/14 - 3*yy**2*yz**2/7 - 3*yy**2*zy**2/7 + 3*yy**2*zz**2/7 + 12*yy*yz*zy*zz/7 - 3*yz**2*zx**2/7 + 3*yz**2*zy**2/7
  D[4][l+  2,l+ 3] =  sqrt(14)*(-xx**3*xz + 3*xx**2*zx*zz + 3*xx*xy**2*xz - 6*xx*xy*zy*zz + 3*xx*xz*zx**2 - 3*xx*xz*zy**2 - 3*xy**2*zx*zz - 6*xy*xz*zx*zy + yx**3*yz - 3*yx**2*zx*zz - 3*yx*yy**2*yz + 6*yx*yy*zy*zz - 3*yx*yz*zx**2 + 3*yx*yz*zy**2 + 3*yy**2*zx*zz + 6*yy*yz*zx*zy)/14
  D[4][l+  2,l+ 4] =  sqrt(7)*(-xx**4 + 6*xx**2*xy**2 + 6*xx**2*zx**2 - 6*xx**2*zy**2 - 24*xx*xy*zx*zy - xy**4 - 6*xy**2*zx**2 + 6*xy**2*zy**2 + yx**4 - 6*yx**2*yy**2 - 6*yx**2*zx**2 + 6*yx**2*zy**2 + 24*yx*yy*zx*zy + yy**4 + 6*yy**2*zx**2 - 6*yy**2*zy**2)/28
  D[4][l+  3,l+-4] =  sqrt(2)*(xx**3*zy + 3*xx**2*xy*zx - 3*xx*xy**2*zy - 3*xx*yx**2*zy - 6*xx*yx*yy*zx + 3*xx*yy**2*zy - xy**3*zx - 3*xy*yx**2*zx + 6*xy*yx*yy*zy + 3*xy*yy**2*zx)/4
  D[4][l+  3,l+-3] =  3*xx**2*xy*zz/4 + 3*xx**2*xz*zy/4 + 3*xx*xy*xz*zx/2 - 3*xx*yx*yy*zz/2 - 3*xx*yx*yz*zy/2 - 3*xx*yy*yz*zx/2 - xy**3*zz/4 - 3*xy**2*xz*zy/4 - 3*xy*yx**2*zz/4 - 3*xy*yx*yz*zx/2 + 3*xy*yy**2*zz/4 + 3*xy*yy*yz*zy/2 - 3*xz*yx**2*zy/4 - 3*xz*yx*yy*zx/2 + 3*xz*yy**2*zy/4
  D[4][l+  3,l+-2] =  sqrt(14)*(-xx**3*zy - 3*xx**2*xy*zx - 3*xx*xy**2*zy + 12*xx*xy*xz*zz + 6*xx*xz**2*zy + 3*xx*yx**2*zy + 6*xx*yx*yy*zx + 3*xx*yy**2*zy - 12*xx*yy*yz*zz - 6*xx*yz**2*zy - xy**3*zx + 6*xy*xz**2*zx + 3*xy*yx**2*zx + 6*xy*yx*yy*zy - 12*xy*yx*yz*zz + 3*xy*yy**2*zx - 6*xy*yz**2*zx - 12*xz*yx*yy*zz - 12*xz*yx*yz*zy - 12*xz*yy*yz*zx)/28
  D[4][l+  3,l+-1] =  sqrt(7)*(-3*xx**2*xy*zz - 3*xx**2*xz*zy - 6*xx*xy*xz*zx + 6*xx*yx*yy*zz + 6*xx*yx*yz*zy + 6*xx*yy*yz*zx - 3*xy**3*zz - 9*xy**2*xz*zy + 12*xy*xz**2*zz + 3*xy*yx**2*zz + 6*xy*yx*yz*zx + 9*xy*yy**2*zz + 18*xy*yy*yz*zy - 12*xy*yz**2*zz + 4*xz**3*zy + 3*xz*yx**2*zy + 6*xz*yx*yy*zx + 9*xz*yy**2*zy - 24*xz*yy*yz*zz - 12*xz*yz**2*zy)/28
  D[4][l+  3,l+ 0] =  sqrt(70)*(3*xx**3*zx + 3*xx**2*xy*zy - 12*xx**2*xz*zz + 3*xx*xy**2*zx - 12*xx*xz**2*zx - 9*xx*yx**2*zx - 6*xx*yx*yy*zy + 24*xx*yx*yz*zz - 3*xx*yy**2*zx + 12*xx*yz**2*zx + 3*xy**3*zy - 12*xy**2*xz*zz - 12*xy*xz**2*zy - 3*xy*yx**2*zy - 6*xy*yx*yy*zx - 9*xy*yy**2*zy + 24*xy*yy*yz*zz + 12*xy*yz**2*zy + 8*xz**3*zz + 12*xz*yx**2*zz + 24*xz*yx*yz*zx + 12*xz*yy**2*zz + 24*xz*yy*yz*zy - 24*xz*yz**2*zz)/140
  D[4][l+  3,l+ 1] =  sqrt(7)*(-3*xx**3*zz - 9*xx**2*xz*zx - 3*xx*xy**2*zz - 6*xx*xy*xz*zy + 12*xx*xz**2*zz + 9*xx*yx**2*zz + 18*xx*yx*yz*zx + 3*xx*yy**2*zz + 6*xx*yy*yz*zy - 12*xx*yz**2*zz - 3*xy**2*xz*zx + 6*xy*yx*yy*zz + 6*xy*yx*yz*zy + 6*xy*yy*yz*zx + 4*xz**3*zx + 9*xz*yx**2*zx + 6*xz*yx*yy*zy - 24*xz*yx*yz*zz + 3*xz*yy**2*zx - 12*xz*yz**2*zx)/28
  D[4][l+  3,l+ 2] =  sqrt(14)*(-xx**3*zx + 3*xx**2*xz*zz + 3*xx*xz**2*zx + 3*xx*yx**2*zx - 6*xx*yx*yz*zz - 3*xx*yz**2*zx + xy**3*zy - 3*xy**2*xz*zz - 3*xy*xz**2*zy - 3*xy*yy**2*zy + 6*xy*yy*yz*zz + 3*xy*yz**2*zy - 3*xz*yx**2*zz - 6*xz*yx*yz*zx + 3*xz*yy**2*zz + 6*xz*yy*yz*zy)/14
  D[4][l+  3,l+ 3] =  xx**3*zz/4 + 3*xx**2*xz*zx/4 - 3*xx*xy**2*zz/4 - 3*xx*xy*xz*zy/2 - 3*xx*yx**2*zz/4 - 3*xx*yx*yz*zx/2 + 3*xx*yy**2*zz/4 + 3*xx*yy*yz*zy/2 - 3*xy**2*xz*zx/4 + 3*xy*yx*yy*zz/2 + 3*xy*yx*yz*zy/2 + 3*xy*yy*yz*zx/2 - 3*xz*yx**2*zx/4 + 3*xz*yx*yy*zy/2 + 3*xz*yy**2*zx/4
  D[4][l+  3,l+ 4] =  sqrt(2)*(xx**3*zx - 3*xx**2*xy*zy - 3*xx*xy**2*zx - 3*xx*yx**2*zx + 6*xx*yx*yy*zy + 3*xx*yy**2*zx + xy**3*zy + 3*xy*yx**2*zy + 6*xy*yx*yy*zx - 3*xy*yy**2*zy)/4
  D[4][l+  4,l+-4] =  xx**3*xy/2 - 3*xx**2*yx*yy/2 - xx*xy**3/2 - 3*xx*xy*yx**2/2 + 3*xx*xy*yy**2/2 + 3*xy**2*yx*yy/2 + yx**3*yy/2 - yx*yy**3/2
  D[4][l+  4,l+-3] =  sqrt(2)*(3*xx**2*xy*xz - 3*xx**2*yy*yz - 6*xx*xy*yx*yz - 6*xx*xz*yx*yy - xy**3*xz + 3*xy**2*yy*yz - 3*xy*xz*yx**2 + 3*xy*xz*yy**2 + 3*yx**2*yy*yz - yy**3*yz)/4
  D[4][l+  4,l+-2] =  sqrt(7)*(-xx**3*xy + 3*xx**2*yx*yy - xx*xy**3 + 6*xx*xy*xz**2 + 3*xx*xy*yx**2 + 3*xx*xy*yy**2 - 6*xx*xy*yz**2 - 12*xx*xz*yy*yz + 3*xy**2*yx*yy - 12*xy*xz*yx*yz - 6*xz**2*yx*yy - yx**3*yy - yx*yy**3 + 6*yx*yy*yz**2)/14
  D[4][l+  4,l+-1] =  sqrt(14)*(-3*xx**2*xy*xz + 3*xx**2*yy*yz + 6*xx*xy*yx*yz + 6*xx*xz*yx*yy - 3*xy**3*xz + 9*xy**2*yy*yz + 4*xy*xz**3 + 3*xy*xz*yx**2 + 9*xy*xz*yy**2 - 12*xy*xz*yz**2 - 12*xz**2*yy*yz - 3*yx**2*yy*yz - 3*yy**3*yz + 4*yy*yz**3)/28
  D[4][l+  4,l+ 0] =  sqrt(35)*(3*xx**4 + 6*xx**2*xy**2 - 24*xx**2*xz**2 - 18*xx**2*yx**2 - 6*xx**2*yy**2 + 24*xx**2*yz**2 - 24*xx*xy*yx*yy + 96*xx*xz*yx*yz + 3*xy**4 - 24*xy**2*xz**2 - 6*xy**2*yx**2 - 18*xy**2*yy**2 + 24*xy**2*yz**2 + 96*xy*xz*yy*yz + 8*xz**4 + 24*xz**2*yx**2 + 24*xz**2*yy**2 - 48*xz**2*yz**2 + 3*yx**4 + 6*yx**2*yy**2 - 24*yx**2*yz**2 + 3*yy**4 - 24*yy**2*yz**2 + 8*yz**4)/280
  D[4][l+  4,l+ 1] =  sqrt(14)*(-3*xx**3*xz + 9*xx**2*yx*yz - 3*xx*xy**2*xz + 6*xx*xy*yy*yz + 4*xx*xz**3 + 9*xx*xz*yx**2 + 3*xx*xz*yy**2 - 12*xx*xz*yz**2 + 3*xy**2*yx*yz + 6*xy*xz*yx*yy - 12*xz**2*yx*yz - 3*yx**3*yz - 3*yx*yy**2*yz + 4*yx*yz**3)/28
  D[4][l+  4,l+ 2] =  sqrt(7)*(-xx**4 + 6*xx**2*xz**2 + 6*xx**2*yx**2 - 6*xx**2*yz**2 - 24*xx*xz*yx*yz + xy**4 - 6*xy**2*xz**2 - 6*xy**2*yy**2 + 6*xy**2*yz**2 + 24*xy*xz*yy*yz - 6*xz**2*yx**2 + 6*xz**2*yy**2 - yx**4 + 6*yx**2*yz**2 + yy**4 - 6*yy**2*yz**2)/28
  D[4][l+  4,l+ 3] =  sqrt(2)*(xx**3*xz - 3*xx**2*yx*yz - 3*xx*xy**2*xz + 6*xx*xy*yy*yz - 3*xx*xz*yx**2 + 3*xx*xz*yy**2 + 3*xy**2*yx*yz + 6*xy*xz*yx*yy + yx**3*yz - 3*yx*yy**2*yz)/4
  D[4][l+  4,l+ 4] =  xx**4/8 - 3*xx**2*xy**2/4 - 3*xx**2*yx**2/4 + 3*xx**2*yy**2/4 + 3*xx*xy*yx*yy + xy**4/8 + 3*xy**2*yx**2/4 - 3*xy**2*yy**2/4 + yx**4/8 - 3*yx**2*yy**2/4 + yy**4/8

  if lmax > 4:
      raise NotImplementedError(f'Too a big {lmax=}')

  return D


def Dmatrix_for_z(z, lmax, order='xyz'):
    return Dmatrix(new_xy_axis(z), lmax, order)

