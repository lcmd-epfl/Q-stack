import numpy as np

def do_fps(x, d=0):
    """Perform Farthest Point Sampling on a set of points.

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
