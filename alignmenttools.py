import numpy as np

def dist_euclidean(X, Y, i, j):
    """
    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    i: int
        Index of point in first point cloud
    j: int
        Index of point in second point cloud
    
    Returns
    -------
    float: The distance between the two points
    """
    return np.sqrt(np.sum((X[i, :] - Y[j, :])**2))

def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between X and Y

    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    
    Returns
    -------
    D: ndarray(M, N)
        The cross-similarity matrix
    
    """
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def get_ssm(X):
    """
    Return the SSM between all rows of a time-ordered Euclidean point cloud X

    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    
    Returns
    -------
    D: ndarray(M, M)
        The self-similarity matrix
    """
    return get_csm(X, X)

def get_path_cost(X, Y, path):
    """
    Return the cost of a warping path that matches two Euclidean 
    point clouds

    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    P1: ndarray(K, 2)
        Warping path
    
    Returns
    -------
    cost: float
        The sum of the Euclidean distances along the warping path 
        between X and Y
    """
    x = X[path[:, 0], :]
    y = Y[path[:, 1], :]
    ds = np.sqrt(np.sum((x-y)**2, 1))
    return np.sum(ds)

def make_path_strictly_increase(path):
    """
    Given a warping path, remove all rows that do not
    strictly increase from the row before
    """
    toKeep = np.ones(path.shape[0])
    i0 = 0
    for i in range(1, path.shape[0]):
        if np.abs(path[i0, 0] - path[i, 0]) >= 1 and np.abs(path[i0, 1] - path[i, 1]) >= 1:
            i0 = i
        else:
            toKeep[i] = 0
    return path[toKeep == 1, :]

def refine_warping_path(path):
    """
    An implementation of the technique in section 4 of 
    "Refinement Strategies for Music Synchronization" by Ewert and Müller

    Parameters
    ----------
    path: ndarray(K, 2)
        A warping path
    
    Returns
    -------
    path_refined: ndarray(N >= K, 2)
        A refined set of correspondences
    """
    N = path.shape[0]
    ## Step 1: Identify all vertical and horizontal segments
    vert_horiz = []
    i = 0
    while i < N-1:
        if path[i+1, 0] == path[i, 0]:
            # Vertical line
            j = i+1
            while path[j, 0] == path[i, 0] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j})
                i = j
        elif path[i+1, 1] == path[i, 1]:
            # Horizontal line
            j = i+1
            while path[j, 1] == path[i, 1] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j})
                i = j
        else:
            i += 1
    
    ## Step 2: Compute local densities
    xidx = []
    density = []
    i = 0
    vhi = 0
    while i < N:
        inext = i+1
        if vhi < len(vert_horiz) and vert_horiz[vhi]['i'] == i: # This is a vertical or horizontal segment
            v = vert_horiz[vhi]
            n_seg = v['j']-v['i']+1
            xidxi = []
            densityi = []
            n_seg_prev = 0
            n_seg_next = 0
            if vhi > 0:
                v2 = vert_horiz[vhi-1]
                if i == v2['j']:
                    # First segment is at a corner
                    n_seg_prev = v2['j']-v2['i']+1
            if vhi < len(vert_horiz) - 1:
                v2 = vert_horiz[vhi+1]
                if v['j'] == v2['i']:
                    # Last segment is a corner
                    n_seg_next = v2['j']-v2['i']+1
            # Case 1: Vertical Segment
            if v['type'] == 'vert':
                xidxi = [path[i, 0] + k/n_seg for k in range(n_seg+1)]
                densityi = [n_seg]*(n_seg+1)
                if n_seg_prev > 0:
                    densityi[0] = n_seg/n_seg_prev
                if n_seg_next > 0:
                    densityi[-2] = n_seg/n_seg_next
                    densityi[-1] = n_seg/n_seg_next
                    inext = v['j']
                else:
                    inext = v['j']+1
            # Case 2: Horizontal Segment
            else:  
                xidxi = [path[i, 0] + k for k in range(n_seg)]
                densityi = [1/n_seg]*n_seg
                if n_seg_prev > 0:
                    xidxi = xidxi[1::]
                    densityi = densityi[1::]
                if n_seg_next > 0:
                    inext = v['j']
                else:
                    inext = v['j']+1
            xidx += xidxi
            density += densityi
            vhi += 1
        else:
            # This is a diagonal segment
            xidx += [path[i, 0], path[i, 0]+1]
            density += [1, 1]
        i = inext
    
    ## Step 3: Integrate densities
    xidx = np.array(xidx)
    density = np.array(density)
    path_refined = [[0, 0]]
    j = 0
    for i in range(1, xidx.size):
        if xidx[i] > xidx[i-1]:
            j += (xidx[i]-xidx[i-1])*density[i-1]
            path_refined.append([xidx[i], j])
    path_refined = np.array(path_refined)
    return path_refined

def downsample_trajectory(X, fac=2):
    """
    Downsample a time-ordered Euclidean trajectory 
    safely with anti-aliasing

    Parameters
    ----------
    X: ndarray(N, d)
        A time-ordered point cloud in d-dimensional Euclidean space
    fac: int
        Factor by which to downsample

    Returns
    ndarray(floor(N/fac), d)
        The downsampled time series
    """
    from skimage.transform import resize
    return resize(X, (int(X.shape[0]/2), X.shape[1]), anti_aliasing=True)