try:
    import sklearn
    del sklearn
except ImportError:
    import warnings
    warnings.warn("\n\nCannot import scikit-learn. Have you installed qstack with the \"regression\" option?\n\n\n\
(for instance, with `pip install qstack[regression] or `pip install qstack[all]``)\n\n", stacklevel=2)
from types import SimpleNamespace
import numpy as np
from qstack.tools import FrozenKeysDict


_global_kernels_dict_names = ['avg', 'rem']
_local_kernels_dict_names = ['G', 'L', 'dot', 'cosine', 'G_sklearn', 'G_custom_c', 'L_sklearn', 'L_custom_c', 'L_custom_py', 'myG', 'myL', 'myLfast']

class GlobalKernelsDict(FrozenKeysDict):
    def __init__(self, dictionary=None):
        super().__init__(_global_kernels_dict_names, dictionary)

class LocalKernelsDict(FrozenKeysDict):
    def __init__(self, dictionary=None):
        super().__init__(_local_kernels_dict_names, dictionary)

defaults = SimpleNamespace(
  sigma=32.0,
  eta=1e-5,
  kernel='L',
  gkernel=None,
  gdict={'alpha':1.0, 'normalize':1, 'verbose':0},
  test_size=0.2,
  n_rep=5,
  splits=5,
  train_size=[0.125, 0.25, 0.5, 0.75, 1.0],
  etaarr=np.logspace(-10, 0, 5).tolist(),
  sigmaarr=np.logspace(0,6, 13).tolist(),
  sigmaarr_mult=np.logspace(0,2, 5).tolist(),
  random_state=0,
  )
