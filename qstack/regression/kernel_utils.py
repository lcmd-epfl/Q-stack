import numpy
import sysconfig
from types import SimpleNamespace
import qstack

defaults = SimpleNamespace(
  sigma=32.0,
  eta=1e-5,
  kernel='L',
  gkernel='avg',
  test_size=0.2,
  n_rep=5,
  splits=5,
  train_size=[0.125, 0.25, 0.5, 0.75, 1.0],
  etaarr=list(numpy.logspace(-10, 0, 5)),
  sigmaarr=list(numpy.logspace(0,6, 13)),
  sigmaarr_mult=list(numpy.logspace(0,2, 5))
  )




def mol_to_dict(mol, species):
    mol_dict = {s:[]  for s in species}
    for a in mol:
        mol_dict[a[0]].append(a[1])
    for k in mol_dict.keys():
        mol_dict[k] = numpy.array(mol_dict[k])
    return mol_dict

def avg_kernel(kernel):
    avg = numpy.sum(kernel) / (kernel.shape[0] * kernel.shape[1])
    return avg


def rematch_kernel(kernel, alpha=0.1):
    from  dscribe.kernels import REMatchKernel as rematchkernel
    rem = rematchkernel(alpha=alpha)
    K_rem = rem.get_global_similarity(kernel)
    return K_rem

def get_gkernel(gkernel):
    if gkernel == 'avg':
        return avg_kernel
    elif gkernel == 'rem':
        return rematch_kernel

def get_covariance(mol1, mol2, max_sizes, kernel , sigma=None):
    from qstack.regression.kernel_utils import get_kernel
    species = sorted(max_sizes.keys())
    mol1_dict = mol_to_dict(mol1, species)
    mol2_dict = mol_to_dict(mol2, species)
    max_size = sum(max_sizes.values())
    K_covar = numpy.zeros((max_size, max_size))
    idx = 0
    for s in species:
        n1 = len(mol1_dict[s])
        n2 = len(mol2_dict[s])
        s_size = max_sizes[s]
        if n1 == 0 or n2 == 0:
            idx += s_size
            continue
        x1 = numpy.pad(mol1_dict[s], ((0, s_size - n1),(0,0)), 'constant')
        x2 = numpy.pad(mol2_dict[s], ((0, s_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+s_size, idx:idx+s_size] = kernel(x1, x2,  sigma)
        idx += s_size
    return K_covar


def get_global_K(X, Y, sigma, local_kernel, global_kernel):
    n_x = len(X)
    n_y = len(Y)
    species = sorted(list(set([s[0] for m in numpy.concatenate((X, Y), axis=0) for s in m])))

    mol_counts = []
    for m in numpy.concatenate((X, Y), axis=0):
        count_species = {s:0 for s in species}
        for a in m:
            count_species[a[0]] += 1
        mol_counts.append(count_species)

    max_atoms = {s:0 for s in species}
    for m in mol_counts:
        for k, v in m.items():
            max_atoms[k] = max([v, max_atoms[k]])
    max_size = sum(max_atoms.values())
    print(max_atoms, max_size, flush=True)
    K_global = numpy.zeros((n_x, n_y))
    print("Computing global kernel elements:\n[", sep='', end='', flush=True)
    for m in range(0, n_x):
        for n in range(0, n_y):
            K_pair = get_covariance(X[m], Y[n], max_atoms, local_kernel, sigma=sigma)
            K_global[m][n] = global_kernel(K_pair)
        if ((m+1) / len(X) * 100)%10 == 0:
            print(f"##### {(m+1) / len(X) * 100}% #####", sep='', end='', flush=True)
    print("]", flush=True)
    print(f"Final global kernel has size : {K_global.shape}", flush=True)
    return K_global

def my_laplacian_kernel(X, Y, gamma):
  """ Compute Laplacian kernel between X and Y """
  def cdist(X, Y):
    K = numpy.zeros((len(X),len(Y)))
    for i,x in enumerate(X):
      x = numpy.array([x] * len(Y))
      d = numpy.abs(x-Y)
      while len(d.shape)>1:
        d = numpy.sum(d, axis=1) # several axis available for np > 1.7.0
      K[i,:] = d
    return K
  K = -gamma * cdist(X, Y)
  numpy.exp(K, K)
  return K

def my_laplacian_kernel_c(X, Y, gamma):
  import os,sys
  import ctypes
  array_2d_double = numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags='CONTIGUOUS')
  try:
      manh = ctypes.cdll.LoadLibrary(qstack.regression.__path__[0]+"/lib/manh.so")
  except:
      manh = ctypes.cdll.LoadLibrary(qstack.regression.__path__[0]+"/lib/manh"+sysconfig.get_config_var('EXT_SUFFIX'))
  manh.manh.restype = ctypes.c_int
  manh.manh.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    array_2d_double,
    array_2d_double,
    array_2d_double]

  K = numpy.zeros((len(X),len(Y)))
  manh.manh(len(X), len(Y), len(X[0]), X, Y, K)
  K *= -gamma
  numpy.exp(K, K)
  return K




def get_kernel(arg, arg2=None):
  """ Returns the kernel function depending on the cli argument """

  if arg=='G':
      from sklearn.metrics.pairwise import rbf_kernel
      local_kernel = rbf_kernel
  elif arg=='L':
      from sklearn.metrics.pairwise import laplacian_kernel
      local_kernel = laplacian_kernel
  elif arg=='myL':
      local_kernel = my_laplacian_kernel
  elif arg=='myLfast':
      local_kernel = my_laplacian_kernel_c

  if arg2 == None:
      return local_kernel
  else:
      if arg2 == 'avg':
          return lambda x, y, s: get_global_K(x, y, s, local_kernel, avg_kernel)
      elif arg2 == 'rem':
          return lambda x, y, s: get_global_K(x, y, s, local_kernel, rematch_kernel)
