import numpy as np

def pca(points, num_components=2):
    """
    Return vectors that lie in the plane that minimizes the orthogonal distance
    from the plane to a point cloud.

    Arguments:
        points : An array of coordinates (size: number of coordinates x 3)
        num_components (optional) : Number of components to keep.

    Returns:
        p1, p2, p3 : Principal components ordered by magnitude.

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

