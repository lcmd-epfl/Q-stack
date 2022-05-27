import numpy
from types import SimpleNamespace

defaults = SimpleNamespace(
  sigma=32.0,
  eta=1e-5,
  kernel='L',
  test_size=0.2,
  n_rep=5,
  splits=5,
  train_size=[0.125, 0.25, 0.5, 0.75, 1.0],
  etaarr=list(numpy.logspace(-10, 0, 5)),
  sigmaarr=list(numpy.logspace(0,6, 13))
  )

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

