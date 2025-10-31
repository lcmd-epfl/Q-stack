import numpy as np


def _Rz(a):
    """Computes the rotation matrix around absolute z-axis.

    Args:
        a (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = np.cos(a)
    A[0,1] = -np.sin(a)
    A[0,2] = 0
    A[1,0] = np.sin(a)
    A[1,1] = np.cos(a)
    A[1,2] = 0
    A[2,0] = 0
    A[2,1] = 0
    A[2,2] = 1
    return A


def _Ry(b):
    """Computes the rotation matrix around absolute y-axis.

    Args:
        b (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = np.cos(b)
    A[0,1] = 0
    A[0,2] = np.sin(b)
    A[1,0] = 0
    A[1,1] = 1
    A[1,2] = 0
    A[2,0] = -np.sin(b)
    A[2,1] = 0
    A[2,2] = np.cos(b)
    return A


def _Rx(g):
    """Computes the rotation matrix around absolute x-axis.

    Args:
        g (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """

    A = np.zeros((3,3))
    A[0,0] = 1
    A[0,1] = 0
    A[0,2] = 0
    A[1,0] = 0
    A[1,1] = np.cos(g)
    A[1,2] = -np.sin(g)
    A[2,0] = 0
    A[2,1] = np.sin(g)
    A[2,2] = np.cos(g)
    return A


def rotate_euler(a, b, g, rad=False):
    """Computes the rotation matrix given Euler angles.

    Args:
        a (float): Alpha Euler angle.
        b (float): Beta Euler angle.
        g (float): Gamma Euler angle.
        rad (bool): Whether the angles are in radians. Defaults to False (degrees).

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """

    if not rad:
        a = a * np.pi / 180
        b = b * np.pi / 180
        g = g * np.pi / 180

    A = _Rz(a)
    B = _Ry(b)
    G = _Rx(g)

    return A@B@G
