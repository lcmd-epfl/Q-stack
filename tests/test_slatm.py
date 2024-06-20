import os
import glob
import numpy as np
from qstack.qml import slatm


def test_slatm_global():
    path = os.path.dirname(os.path.realpath(__file__))
    v0 = np.load(f'{path}/data/slatm/slatm_global.npy')
    xyzs = sorted(glob.glob(f"{path}/data/slatm/*.xyz"))
    v = slatm.get_slatm_for_dataset(xyzs, progress=False, global_repr=True)
    assert(np.linalg.norm(v-v0)<1e-10)


def test_slatm_local():
    path = os.path.dirname(os.path.realpath(__file__))
    v0 = np.load(f'{path}/data/slatm/slatm_local.npy')
    xyzs = sorted(glob.glob(f"{path}/data/slatm/*.xyz"))
    v = slatm.get_slatm_for_dataset(xyzs, progress=False)
    assert(np.linalg.norm(v-v0)<1e-10)


if __name__ == '__main__':
    test_slatm_local()
    test_slatm_global()
