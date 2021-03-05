import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from alignmenttools import dist_euclidean, downsample_trajectory

def dtw(X, Y):
    """
    Perform dynamic time warping between two
    Euclidean trajectories

    Parameters
    ----------
    X: ndarray(M, d)
        An Mxd array of coordinates for the first path
    Y: ndarray(N, d)
        An Nxd array of coordinates for the second path
    
    Returns
    -------
    path: list of [i, j]
        The warping path to align X and Y
    """
    M = X.shape[0]
    N = Y.shape[0]
    S = np.zeros((M, N)) # The dynamic programming matrix
    S[0, 0] = dist_euclidean(X, Y, 0, 0) # The stopping condition / base case
    ## TODO: Fill this in.  At the end, you should return
    ## an optimal warping path, expressed as a list of [i, j]
    return []


def create_mask(M, N, path, radius):
    """
    Fill a square block with values

    Parameters
    ----------
    M: int
        Number of points in the first trajectory
    N: int
        Number of points in the second trajectory
    p: list of [i, j]
        A warping path one level up
    radius: int
        Half the width of the box to place around [2*i, 2*j]
        for each [i, j] one level up
    
    Returns
    -------
    An MxN sparse array which has a 1 in every cell
    that needs to be checked and a 0 elsewhere
    """
    Occ = sparse.lil_matrix((M, N))
    ## TODO: Fill this in; loop through all of the elements
    ## in the path and place a box around each one
    return Occ

def get_mask_indices_inorder(Occ):
    """
    Parameters
    ----------
    Occ: scipy.sparse
        An MxN array of occupied cells to visit
        in a dynamic programming problem
    
    Returns
    -------
    List of [i, j]: A list of coordinates in the order
    that they need to be filled to satisfy dependencies
    """
    I, J = Occ.nonzero()
    # Sort cells in raster order
    idx = np.argsort(J)
    I = I[idx]
    J = J[idx]
    idx = np.argsort(I, kind='stable')
    I = I[idx]
    J = J[idx]
    ret = np.array([I, J], dtype=int).T
    return ret.tolist()


def fastdtw(X, Y, radius, level = 0, do_plot=False):
    """
    An implementation of [1]
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador and Philip Chan
    
    Parameters
    ----------
    X: ndarray(M, d)
        An Mxd array of coordinates for the first path
    Y: ndarray(N, d)
        An Nxd array of coordinates for the second path
    radius: int
        Radius of the l-infinity box that determines sparsity structure
        at each level
    level: int
        An int for keeping track of the level of recursion
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files
    
    Returns
    -------
    path: list of [i, j]
        The warping path to align X and Y at this level
    """
    M = X.shape[0]
    N = Y.shape[0]
    S = sparse.lil_matrix((M, N)) # Matrix for storing the cumulative cost
    
    ## TODO: Fill this in

    path = [[0, 0]] ## TODO: This is a dummy value
    
    if do_plot: # pragma: no cover
        plt.figure(figsize=(8, 8))
        plt.imshow(S.toarray())
        P = np.array(path)
        plt.scatter(P[:, 1], P[:, 0], c='C1')
        plt.title("Level {}".format(level))
        plt.savefig("%i.png"%level, bbox_inches='tight')

    return path