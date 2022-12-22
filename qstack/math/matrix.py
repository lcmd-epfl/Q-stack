import numpy as np


def sqrtm(m, eps=1e-13):
    e, b = np.linalg.eigh(m)
    e[abs(e) < eps] = 0.0
    sm = b @ np.diag(np.sqrt(e)) @ b.T
    return (sm+sm.T)*0.5
