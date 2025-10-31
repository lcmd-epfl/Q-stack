import numpy as np


def do_fps(x, d=0):
    """Perform Farthest Point Sampling on a set of points.

    Dral P O, Owens A, Yurchenko S N and Thiel W 2017 J. Chem. Phys. 146 244108 doi:10.1063/1.4989536
    Imbalzano G, Anelli A, Giofré D, Klees S, Behler J and Ceriotti M 2018 J. Chem. Phys. 148 241730 doi:10.1063/1.5024611
    Rossi K, Jurásková V, Wischert R, Garel L, Corminboeuf C and Ceriotti M 2020 J. Chem. Theory Comput. 16 5139–49 doi:10.1021/acs.jctc.0c00362

    Code from Giulio Imbalzano.

    Args:
        x (numpy.ndarray): 2D array of points, shape (n_points, n_features).
        d (int): Number of points to sample. If 0, samples all points. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - iy (numpy.ndarray): Indices of sampled points.
            - measure (numpy.ndarray): Distances to nearest selected point for each iteration.
    """
    n = len(x)
    if d==0:
        d = n
    iy = np.zeros(d,int)
    measure = np.zeros(d-1,float)
    iy[0] = 0
    # Faster evaluation of Euclidean distance
    n2 = np.sum(x*x, axis=1)
    dl = n2 + n2[iy[0]] - 2.0*np.dot(x,x[iy[0]])
    for i in range(1,d):
        iy[i], measure[i-1] = np.argmax(dl), np.amax(dl)
        nd = n2 + n2[iy[i]] - 2.0*np.dot(x,x[iy[i]])
        dl = np.minimum(dl,nd)
    return iy, measure
