#!/usr/bin/env python3

import numpy as np
from qstack.regression import kernel, local_kernels


def test_local_kernels():
    #np.random.seed(666)
    #X = np.random.rand(2,4)
    #Y = np.random.rand(2,4)
    #K_G_good = np.zeros((len(X),len(Y)))
    #K_L_good = np.zeros((len(X),len(Y)))
    #for i, x in enumerate(X):
    #    for j, y in enumerate(Y):
    #        K_G_good[i,j] = np.dot(x-y, x-y)
    #        K_L_good[i,j] = np.sum(abs(x-y))
    #np.exp(-K_G_good/2, out=K_G_good)
    #np.exp(-K_L_good/2, out=K_L_good)
    #K_dot_good = np.dot(X, Y.T)
    #K_cos_good = K_dot_good / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))

    X = np.array([[0.70043712, 0.84418664, 0.67651434, 0.72785806], [0.95145796, 0.0127032 , 0.4135877 , 0.04881279]])
    Y = np.array([[0.09992856, 0.50806631, 0.20024754, 0.74415417], [0.192892  , 0.70084475, 0.29322811, 0.77447945]])
    K_G_good = np.array([[0.70444747, 0.80765894], [0.47248452, 0.45157228]])
    K_L_good = np.array([[0.48938983, 0.58251676], [0.32374891, 0.31778924]])
    K_dot_good = np.array([[1.1760054, 1.48883663], [0.22067605, 0.35151164]])
    K_cos_good = np.array([[0.85579088, 0.91287375], [0.22883447, 0.30712225]])

    for akernel in ['G', 'G_sklearn', 'G_custom_c']:
        K = kernel.kernel(X, Y, akernel=akernel, sigma=2.0)
        assert np.allclose(K, K_G_good)

    for akernel in ['L', 'L_sklearn', 'L_custom_c', 'L_custom_py']:
        K = kernel.kernel(X, Y, akernel=akernel, sigma=2.0)
        assert np.allclose(K, K_L_good)

    K = kernel.kernel(X.reshape((2,2,2)), Y.reshape((2,2,2)), akernel='L_custom_py', sigma=2.0)
    assert np.allclose(K, K_L_good)

    K = kernel.kernel(X, Y, akernel='dot')
    assert np.allclose(K, K_dot_good)

    K = kernel.kernel(X, Y, akernel='cosine')
    assert np.allclose(K, K_cos_good)


def test_batched_local_kernels():
    X = np.array([[0.70043712, 0.84418664, 0.67651434, 0.72785806], [0.95145796, 0.0127032 , 0.4135877 , 0.04881279]])
    Y = np.array([[0.09992856, 0.50806631, 0.20024754, 0.74415417], [0.192892  , 0.70084475, 0.29322811, 0.77447945]])
    K_L_good = np.array([[0.48938983, 0.58251676], [0.32374891, 0.31778924]])

    X_huge = np.concatenate([X]*1_000, axis=1)
    X_huge = np.concatenate([X_huge]*1000, axis=0)
    Y_huge = np.concatenate([Y]*1_000, axis=1)
    Y_huge = np.concatenate([Y_huge]*50, axis=0)
    K_L_good_huge = np.concatenate([K_L_good]*1000, axis=0)
    K_L_good_huge = np.concatenate([K_L_good_huge]*50, axis=1)

    local_kernels.RAM_BATCHING_SIZE = 1024**2 * 50  # 50MiB
    for akernel in ['L_custom_c', 'L_custom_py', 'L', 'L_sklearn']:
        K = kernel.kernel(X_huge, Y_huge, akernel=akernel, sigma=2.0*1000)
        assert np.allclose(K, K_L_good_huge)

if __name__ == '__main__':
    test_local_kernels()
    test_batched_local_kernels()
