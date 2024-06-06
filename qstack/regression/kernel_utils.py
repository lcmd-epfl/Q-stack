import argparse
import numpy as np
from types import SimpleNamespace
import qstack


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, defaults.gdict)
        for value in values:
            key, value = value.split('=')
            for t in [int, float]:
                try:
                    value = t(value)
                    break
                except:
                    continue
            getattr(namespace, self.dest)[key] = value


defaults = SimpleNamespace(
  sigma=32.0,
  eta=1e-5,
  kernel='L',
  gkernel=None,
  gdict={'alpha':1.0, 'normalize':1},
  test_size=0.2,
  n_rep=5,
  splits=5,
  train_size=[0.125, 0.25, 0.5, 0.75, 1.0],
  etaarr=list(np.logspace(-10, 0, 5)),
  sigmaarr=list(np.logspace(0,6, 13)),
  sigmaarr_mult=list(np.logspace(0,2, 5)),
  random_state=0,
  )


def mol_to_dict(mol, species):
    """

    .. todo::
        Write the docstring
    """

    mol_dict = {s:[]  for s in species}
    for a in mol:
        mol_dict[a[0]].append(a[1])
    for k in mol_dict.keys():
        mol_dict[k] = np.array(mol_dict[k])
    return mol_dict


def avg_kernel(kernel, options):
    """

    .. todo::
        Write the docstring
    """
    avg = np.sum(kernel) / (kernel.shape[0] * kernel.shape[1])
    return avg


def rematch_kernel(kernel, options):
    """

    .. todo::
        Write the docstring
    """
    alpha = options['alpha']
    thresh = 1e-6
    n, m = kernel.shape

    K = np.exp(-(1 - kernel) / alpha)

    en = np.ones((n,)) / float(n)
    em = np.ones((m,)) / float(m)

    u = en.copy()
    v = em.copy()

    niter = 0
    error = 1

    while error > thresh:
        v_prev = v
        u_prev = u

        u = np.divide(en, np.dot(K, v))
        v = np.divide(em, np.dot(K.T, u))

        if niter % 5:
            error = np.sum((u - u_prev) ** 2) / np.sum(u ** 2) + np.sum((v - v_prev) ** 2) / np.sum(v **2)

        niter += 1
    p_alpha = np.multiply(np.multiply(K, u.reshape((-1, 1))) , v)
    K_rem = np.sum(np.multiply(p_alpha, kernel))
    return K_rem


def normalize_kernel(kernel, self_x=None, self_y=None):
    """

    .. todo::
        Write the docstring
    """
    print("Normalizing kernel.")
    if self_x == None and self_y == None:
        self_cov = np.diag(kernel).copy()
        self_x = self_cov
        self_y = self_cov
    for n in range(kernel.shape[0]):
        for m in range(kernel.shape[1]):
            kernel[n][m] /= np.sqrt(self_x[n]*self_y[m])
    return kernel


def get_covariance(mol1, mol2, max_sizes, kernel , sigma=None):
    """

    .. todo::
        Write the docstring
    """
    from qstack.regression.kernel_utils import get_kernel
    species = sorted(max_sizes.keys())
    mol1_dict = mol_to_dict(mol1, species)
    mol2_dict = mol_to_dict(mol2, species)
    max_size = sum(max_sizes.values())
    K_covar = np.zeros((max_size, max_size))
    idx = 0
    for s in species:
        n1 = len(mol1_dict[s])
        n2 = len(mol2_dict[s])
        s_size = max_sizes[s]
        if n1 == 0 or n2 == 0:
            idx += s_size
            continue
        x1 = np.pad(mol1_dict[s], ((0, s_size - n1),(0,0)), 'constant')
        x2 = np.pad(mol2_dict[s], ((0, s_size - n2),(0,0)), 'constant')
        K_covar[idx:idx+s_size, idx:idx+s_size] = kernel(x1, x2,  sigma)
        idx += s_size
    return K_covar


def get_global_K(X, Y, sigma, local_kernel, global_kernel, options):
    """

    .. todo::
        Write the docstring
    """
    n_x = len(X)
    n_y = len(Y)
    species = sorted(list(set([s[0] for m in np.concatenate((X, Y), axis=0) for s in m])))

    mol_counts = []
    for m in np.concatenate((X, Y), axis=0):
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
    K_global = np.zeros((n_x, n_y))
    print("Computing global kernel elements:\n[", sep='', end='', flush=True)
    if X[0] == Y[0]:
        self = True
    else:
        self = False
        self_X = []
        self_Y = []
    for m in range(0, n_x):
        if not self:
            K_self = get_covariance(X[m], X[m], max_atoms, local_kernel, sigma=sigma)
            self_X.append(global_kernel(K_self, options))
        for n in range(0, n_y):
            if not self:
                K_self = get_covariance(Y[m], Y[m], max_atoms, local_kernel, sigma=sigma)
                self_Y.append(global_kernel(K_self, options))
            K_pair = get_covariance(X[m], Y[n], max_atoms, local_kernel, sigma=sigma)
            K_global[m][n] = global_kernel(K_pair, options)
        if ((m+1) / len(X) * 100)%10 == 0:
            print(f"##### {(m+1) / len(X) * 100}% #####", sep='', end='', flush=True)
    print("]", flush=True)
    if options['normalize'] == True:
        if self :
            K_global = normalize_kernel(K_global, self_x=None, self_y=None)
        else:
            K_global = normalize_kernel(K_global, self_x=self_X, self_y=self_Y)
    print(f"Final global kernel has size : {K_global.shape}", flush=True)
    return K_global


def my_laplacian_kernel(X, Y, gamma):
  """ Compute Laplacian kernel between X and Y

  .. todo::
      Write the docstring
  """
  def cdist(X, Y):
    K = np.zeros((len(X),len(Y)))
    for i,x in enumerate(X):
      x = np.array([x] * len(Y))
      d = np.abs(x-Y)
      while len(d.shape)>1:
        d = np.sum(d, axis=1) # several axis available for np > 1.7.0
      K[i,:] = d
    return K
  K = -gamma * cdist(X, Y)
  np.exp(K, K)
  return K


def my_kernel_c(akernel):
    """

    .. todo::
        Write the docstring
    """
    import ctypes
    import sysconfig
    array_2d_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='CONTIGUOUS')
    def kernel_func_c(X, Y, gamma):
        try:
            manh = ctypes.cdll.LoadLibrary(qstack.regression.__path__[0]+"/lib/manh.so")
        except:
            manh = ctypes.cdll.LoadLibrary(qstack.regression.__path__[0]+"/lib/manh"+sysconfig.get_config_var('EXT_SUFFIX'))
        if akernel == 'myLfast':
            kfunc = manh.manh
        elif akernel == 'myG':
            kfunc = manh.eucl
        kfunc.restype = ctypes.c_int
        kfunc.argtypes = [
          ctypes.c_int,
          ctypes.c_int,
          ctypes.c_int,
          array_2d_double,
          array_2d_double,
          array_2d_double]
        K = np.zeros((len(X),len(Y)))
        kfunc(len(X), len(Y), len(X[0]), X, Y, K)
        K *= -gamma
        np.exp(K, K)
        return K
    return kernel_func_c


def get_local_kernel(arg):
    """

    .. todo::
        Write the docstring
    """
    if arg=='G':
        from sklearn.metrics.pairwise import rbf_kernel
        return rbf_kernel
    elif arg=='dot':
        from sklearn.metrics.pairwise import linear_kernel
        return lambda x,y,s: linear_kernel(x, y)
    elif arg=='L':
        from sklearn.metrics.pairwise import laplacian_kernel
        return laplacian_kernel
    elif arg=='myL':
        return my_laplacian_kernel
    elif arg in ['myLfast', 'myG']:
        return my_kernel_c(arg)
    elif arg=='cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        return lambda x,y,s: cosine_similarity(x, y)
    else:
        raise Exception(f'{arg} kernel is not implemented') # TODO


def get_global_kernel(arg, local_kernel):
    """

    .. todo::
        Write the docstring
    """
    gkernel, options = arg
    if gkernel == 'avg':
        global_kernel = avg_kernel
    elif gkernel == 'rem':
        global_kernel = rematch_kernel
    else:
        raise Exception(f'{gkernel} kernel is not implemented') # TODO
    return lambda x, y, s: get_global_K(x, y, s, local_kernel, global_kernel, options)


def get_kernel(arg, arg2=None):
  """ Returns the kernel function depending on the cli argument

  .. todo::
      Write the docstring
  """

  local_kernel = get_local_kernel(arg)

  if arg2 is None or arg2[0] is None:
      return local_kernel
  else:
      return get_global_kernel(arg2, local_kernel)


def train_test_split_idx(y, idx_test=None, idx_train=None,
                         test_size=defaults.test_size, random_state=defaults.random_state):

    from sklearn.model_selection import train_test_split

    if idx_test is None and idx_train is None:
        idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state)
    elif idx_test is not None and idx_train is None:
        idx_train = np.delete(np.arange(len(y)), idx_test)
        # Check there is no repeating indices in `idx_test`.
        # Note that negative indices could be used (`np.delete` handles them)
        # this is why a check if some indices are duplicated is not sufficient.
        assert len(idx_test)+len(idx_train)==len(y)
    elif idx_test is None and idx_train is not None:
        idx_test = np.delete(np.arange(len(y)), idx_train)
        assert len(idx_test)+len(idx_train)==len(y)
    else:
        assert len(np.delete(np.arange(len(y)), idx_train)) == len(y)-len(idx_train)
        assert len(np.delete(np.arange(len(y)), idx_test)) == len(y)-len(idx_test)
        if len(np.delete(np.arange(len(y)), idx_test+idx_train)) != len(y)-len(idx_test)-len(idx_train):
            warnings.warn('Train and test set indices overlap. Is it intended?')
    return idx_train, idx_test, y[idx_train], y[idx_test]
