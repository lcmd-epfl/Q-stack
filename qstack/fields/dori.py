#!/usr/bin/env python3
import numpy as np
import scipy.linalg as spl
import pyscf

def _dori_inner_formula(values, with_color=False):
    """Computes the DORI indicator for a set of points in space, from the data given by pyscf
    Args:
      - values: the evaluation of the density and its derivatives, of shape [N_components, N_points],
                where the components are [ρ, x,y,z, xx, xy, xz, yy, yz, zz]
      

    """
    _explainer = """
    T = N( (N(ρ) / ρ)**2 )
    = N( 1/ρ**2 * (x**2+y**2+z**2) )
    = 1/ρ**2 * (x(x**2)+x(y**2)+x(z**2) ; _; _) + (x**2+y**2+z**2) * N(1/ρ**2)
    = 1/ρ**2 * (2*xx*x+ 2*xy*y + 2*xz*z ; _; _) - (x**2+y**2+z**2) * 2/ρ**3 * (x;y;z)
    = 2/ρ**2 * ((xx*x+xy*y+xz*z; _; _) - (x**2+y**2+z**2)/ρ * (x;y;z) )

    DORI = T**2 / (1/ρ**2 * (x**2+y**2+z**2))**3
    = 4/ρ**4 (
        (xx*x+xy*y+xz*z)**2+ (yx*x+yy*y+yz*z)**2 + (zx*x+zy*y+zz*z)**2
        + (x**2+y**2+z**2)**3/ρ**2
        -2*(x**2+y**2+z**2)/ρ * (xx*x*x+xy*y*x+xz*z*x + yx*x*y+yy*y*y+yz*z*y + zx*x*z+zy*y*z+zz*z*z)
      )
      /
      1/ρ**6 * (x**2+y**2+z**2)**3

    = 4/ρ**6 (
        ρ**2 * ((xx*x+xy*y+xz*z)**2+ (yx*x+yy*y+yz*z)**2 + (zx*x+zy*y+zz*z)**2)
        + (x**2+y**2+z**2)**3
        -2*ρ*(x**2+y**2+z**2) * (xx*x*x+xy*y*x+xz*z*x + yx*x*y+yy*y*y+yz*z*y + zx*x*z+zy*y*z+zz*z*z)
      )
      /
      1/ρ**6 * (x**2+y**2+z**2)**3


    = 4 * (
        ρ**2 * ((xx*x+xy*y+xz*z)**2+ (yx*x+yy*y+yz*z)**2 + (zx*x+zy*y+zz*z)**2)/(x**2+y**2+z**2)**3
        + 1
        -2*ρ/(x**2+y**2+z**2)**2 * (xx*x*x+xy*y*x+xz*z*x + yx*x*y+yy*y*y+yz*z*y + zx*x*z+zy*y*z+zz*z*z)
      )

    """

    rho = values[0]
    first = values[1:4]
    second = np.zeros_like(values, shape=(3, 3, values.shape[-1]))
    second[:,0,:] = values[4:7]
    second[0,1:,:] = values[5:7]
    second[1,1:,:] = values[7:9]
    second[2,1:,:] = values[8:10]

    nabla2 = (first**2).sum(axis=0)  # (x**2+y**2+z**2)
    bigterm= np.einsum('xp,xyp,yp->p', first, second, first)  # (xx*x*x+xy*y*x+xz*z*x + yx*x*y+yy*y*y+yz*z*y + zx*x*z+zy*y*z+zz*z*z)

    final = (np.einsum('xyp,yp->xp', second, first)**2).sum(axis=0)  # ((xx*x+xy*y+xz*z)**2+ (yx*x+yy*y+yz*z)**2 + (zx*x+zy*y+zz*z)**2)
    
    if with_color:
        color = np.empty_like(final)
        for index in range(color.shape[0]):
            color[index] = spl.eigvalsh(second[:,:,index], overwrite_a=True)[1]
    del second
    if with_color:
        color = np.sign(color)
        color *= rho

    final *= rho/nabla2
    final -= 2*bigterm
    del bigterm
    final *= rho/nabla2**2

    needs_redo = np.logical_not(np.isfinite(final))
    if needs_redo.sum() > 0:
        print(len(needs_redo), np.nonzero(needs_redo)[0], final[needs_redo])
    final[needs_redo] = 0


    del nabla2
    final +=1
    final *=4
    if with_color:
        return final, color
    else:
        return final 

def get_dori_from_density(mol, rho_coeffs, coords, with_color=False):
    values_all = mol.eval_ao('GTOval_sph_deriv2', coords)
    rho_evaled = values_all @ rho_coeffs
    del values_all
    return _dori_inner_formula(rho_evaled, with_color)


def get_dori_from_rdm(mol, rdm, coords, with_color=False):
    import time; a=time.perf_counter()
    values_all = mol.eval_ao('GTOval_sph_deriv2', coords)
    print('vaa', time.perf_counter()-a)
    half_done = np.tensordot(values_all, rdm, 1)
    # this asymmetric pre-evaluation assumes the rdm is a symmetric matrix.

    rho_evaled = np.empty_like(values_all, shape=(10,len(coords)))
    np.einsum('px,px->p', values_all[0], half_done[0], out=rho_evaled[0])
    np.einsum('cpx,px->cp', half_done[1:10], values_all[0], out=rho_evaled[1:10])
    second_term = np.empty_like(values_all, shape=(6,len(coords)))
    np.einsum('cpx,px->cp', half_done[1:4], values_all[1], out=second_term[0:3])
    np.einsum('cpx,px->cp', half_done[2:4], values_all[2], out=second_term[3:5])
    np.einsum('px,px->p', half_done[3], values_all[3], out=second_term[5])
    rho_evaled[4:] += second_term
    del second_term, values_all, half_done

    rho_evaled[1:10]*=2
    print('rho', time.perf_counter()-a)
    
    try:
        return _dori_inner_formula(rho_evaled, with_color)
    finally:
        print('end', time.perf_counter()-a)


def get_dori(mol, qty, coords, with_color=False, MAX_SZ=5*1024**3):
    if qty.ndim==1:
        # the inner calculation takes the main evaluation array (10), computes a new second derivative array (9),
        # two temporary variables (2) and a temp in-flight buffer of up to second-derivative-size (9)

        # meanwhile, computing this evaluation array in the first place requires N_ao of temp-buffer.
        # (still, we desirte a margin for the internal calculations done by eval_ao)
        floats_per_point = max(30, 5*mol.nao)
        _func = get_dori_from_density
    elif qty.ndim==2:
        # computing this evaluation array form an rdm uses several arrays (variables and in-flight): 2*N_ao*10 + N_ao*9 + 2
        floats_per_point = max(30, 29*mol.nao+2)
        _func = get_dori_from_rdm

    max_points = MAX_SZ // (floats_per_point * np.dtype(qty.dtype).itemsize)
    dori = np.zeros_like(qty, shape=len(coords))
    if with_color:
        color = np.zeros_like(qty, shape=len(coords))
    print(max_points, len(coords))
    
    for point_begin in range(0, len(coords), max_points):
        point_end = min(point_begin+max_points, len(coords))
        if with_color:
            dori[point_begin:point_end],color[point_begin:point_end] = _func(mol, qty, coords[point_begin:point_end], with_color)
        else:
            dori[point_begin:point_end] = _func(mol, qty, coords[point_begin:point_end], with_color)
    if with_color:
        return dori, color
    else:
        return dori
        


if __name__ == '__main__':
    import sys
    import qstack
    # args: xyzfile, basis, datafile, outpath
    
    mol = qstack.compound.xyz_to_mol(sys.argv[1], basis=sys.argv[2], parse_comment=True)
    cube = pyscf.tools.cubegen.Cube(mol) ; grid = cube.get_coords()
    dori, color = qstack.fields.dori.get_dori(mol, np.load(sys.argv[3]), grid, with_color=True)
    dorimod = dori * (abs(color)<0.03)*(abs(color)> 1E-5)
    cube.write(dori.reshape(80,80,80), sys.argv[4]+'-dori.cube')
    cube.write(dorimod.reshape(80,80,80), sys.argv[4]+'-dori-cropped.cube')
    cube.write(color.reshape(80,80,80), sys.argv[4]+'-color.cube')
