import os
import numpy as np
from qstack.regression import kernel, kernel_utils


## TODO: complete the test for global kernel computations; this is just a copy from test_saphm()

def test_avg_kernel():

    path = os.path.dirname(os.path.realpath(__file__))
    X_dir = os.path.join(path, 'data/SPAHM_a_H2O/')
    mols = [np.load(os.path.join(X_dir, f), allow_pickle=True) for f in os.listdir(X_dir) if os.path.isfile(os.path.join(X_dir,f))]

    K = kernel.kernel(mols, akernel='L', gkernel='avg', sigma=1.0)

    true_K = np.array(  [[1.       ,  0.58784717, 0.58784717], \
                        [0.58784717, 1.         , 0.74535599], \
                        [0.58784717, 0.74535599 , 1.        ]])
    print(K)

    assert(K.shape == (3,3))
    assert(np.abs(np.sum(K-true_K)) < 1e-05)

test_avg_kernel()

