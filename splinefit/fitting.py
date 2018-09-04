import numpy as np

def pca(points, num_components=2):
    """
    Return vectors that lie in the plane that minimizes the orthogonal distance
    from the plane to a point cloud.

    Arguments:
        points : An array of coordinates (size: number of coordinates x 3)
        num_components (optional) : Number of components to keep.

    Returns:
        eig_vec : Principal components ordered by magnitude. `eig_vec[:,0]` is
        the first column vector corresponding to the maximum modulus eigenvalue.

    """

    assert points.shape[1] == 3

    mean = np.sum(points,0)/points.shape[0]

    x = 0*points

    for i in range(3):
        x[:,i] = points[:,i] - mean[i]

    A = x.T.dot(x)

    eig, eig_vec = np.linalg.eig(A)

    # Order eigenvalues in descending order
    min_pca = np.argsort(-eig)

    return eig_vec[:,0:num_components]

def projection(points, basis):
    """
    Project a point cloud onto a plane defined by some basis.

    Arguments:
        points : An array of size num points x q that contains the point cloud.
            Here, `points[0,:]` is the first point with coordinates 
            `x_0, x_1, ..., x_(q-1)`
        basis : Basis to use for the projection. Here, `basis[:,0]` is the first
            basis vector.

    """

    # Normalize the basis in case it is not already normalized
    norms = np.linalg.norm(basis, axis=0)
    b = basis/norms

    out = points.dot(b)
    return out



def normalize(vecs):
    """
    Normalize a collection of vectors:
        
    Arguments:
        vecs : Array of row vectors.  Here, `vecs[0,:]` is the first vector.

    """
    norms = np.linalg.norm(vecs, axis=1)
    norms = np.tile(norms, (vecs.shape[1], 1)).T
    v = vecs/norms
    return v
