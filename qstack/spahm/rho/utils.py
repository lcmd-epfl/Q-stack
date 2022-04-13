import pyscf
import numpy
import resource
import time

def readmol(fin, basis, charge=0, spin=0):
  """ Read xyz and return pyscf-mol object """
  f = open(fin, "r")
  molxyz = '\n'.join(f.read().split('\n')[2:])
  f.close()
  mol = pyscf.gto.Mole()
  mol.atom = molxyz
  mol.basis = basis
  mol.charge = charge
  mol.spin = spin
  mol.build()
  return mol

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

def get_kernel(arg):
  """ Returns the kernel function depending on the cli argument """
  if arg=='G':
    from sklearn.metrics.pairwise import rbf_kernel
    return rbf_kernel
  elif arg=='L':
    from sklearn.metrics.pairwise import laplacian_kernel
    return laplacian_kernel
  elif arg=='myL':
    return my_laplacian_kernel

def compile_repr(X0, lens):
  X = numpy.zeros((len(X0), *max(lens)))
  if len(X.shape)==2:
    for i,x in enumerate(X0):
      X[i,0:lens[i][-1]] = x
  else:
    for i,x in enumerate(X0):
      X[i,:,0:lens[i][-1]] = x
  return X

def unix_time_decorator(func):
# thanks to https://gist.github.com/turicas/5278558
  def wrapper(*args, **kwargs):
    start_time, start_resources = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    ret = func(*args, **kwargs)
    end_resources, end_time = resource.getrusage(resource.RUSAGE_SELF), time.time()
    print(func.__name__, ':  real: %.4f  user: %.4f  sys: %.4f'%
          (end_time - start_time,
           end_resources.ru_utime - start_resources.ru_utime,
           end_resources.ru_stime - start_resources.ru_stime))
    return ret
  return wrapper
