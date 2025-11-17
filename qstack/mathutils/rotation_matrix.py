"""Rotation matrix generation functions."""

import numpy as np


def _Rz(a):
    """Compute the rotation matrix around laboratory z-axis.

    Args:
        a (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [0,   0,  1],
    ])


def _Ry(b):
    """Compute the rotation matrix around laboratory y-axis.

    Args:
        b (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    cb, sb = np.cos(b), np.sin(b)
    return np.array([
        [ cb, 0, sb],
        [ 0,  1, 0 ],
        [-sb, 0, cb],
    ])


def _Rx(g):
    """Compute the rotation matrix around laboratory x-axis.

    Args:
        g (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [1, 0,  0 ],
        [0, cg, -sg],
        [0, sg,  cg],
    ])


def rotate_euler(a, b, g, rad=False):
    """Compute the rotation matrix given Cardan angles (x-y-z).

    Args:
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool): Whether the angles are in radians. Defaults to False (degrees).

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    if not rad:
        a, b, g = np.radians([a, b, g])

    A = _Rz(a)
    B = _Ry(b)
    G = _Rx(g)

    return A@B@G
