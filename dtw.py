import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from alignmenttools import dist_euclidean, downsample_trajectory

def delannoy(M, N):
    """
    Compute the Delannoy number D(M, N) using dynamic programming
    
    Parameters
    ----------
    M: int
        Number of samples in the first time series
    N: int
        Number of samples in the second time series
    
    Returns
    -------
    int: D(M, N)
    """
    D = np.ones((M, N), dtype=int)
    for i in range(1, M):
        for j in range(1, N):
            D[i, j] = D[i-1, j] + D[i, j-1] + D[i-1, j-1]
    return D[-1, -1]

def plot_all_warppaths(M, N, path = [[0, 0]], params = {"num":1}):
    """
    Make plots of all warping paths between two time series of
    specified lengths

    Parameters
    ----------
    M: int
        Number of samples in the first time series
    N: int
        Number of samples in the second time series
    path: list of [i, j]
        Recursively constructed warping path from one time
        series to the next
    params: dict
        Used for keeping track of which warping path we're on
        and how many warping paths total there are
    """
    if not "D" in params:
        params["D"] = delannoy(M, N)
    p = path[-1] # Pull out the last pair in the warping path
    if p[0] == M-1 and p[1] == N-1:
        # Stopping condition: This warping path has reached the end
        plt.clf()
        plt.title("{} x {} Warping Path {} of {}".format(M, N, params["num"], params["D"]))
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0])
        plt.scatter(path[:, 1], path[:, 0], color='k', zorder=10)
        plt.xticks(np.arange(N), ["%i"%i for i in range(N)])
        plt.yticks(np.arange(M), ["%i"%i for i in range(M)])
        plt.gca().invert_yaxis()
        plt.ylabel("First Time Series")
        plt.xlabel("Second Time Series")
        plt.savefig("Path{}.png".format(params["num"]), bbox_inches='tight')
        params["num"] += 1
    else:
        ## TODO: Fill this in
        pass

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