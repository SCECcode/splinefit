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

    mu = mean(points)

    x = 0*points

    for i in range(3):
        x[:,i] = points[:,i] - mu[i]

    A = x.T.dot(x)

    eig, eig_vec = np.linalg.eig(A)

    min_pca = np.argsort(-eig)

    return eig_vec[:,min_pca[0:num_components]]

def mean(points):
    """
    Compute the mean of a collection of points in space.
        
    Arguments
        points : An array of coordinates (size: number of coordinates x q)

    Returns:
        out : mean value of coordinate (size: q)

    """
    out = np.sum(points,0)/points.shape[0]
    return out

def normalize(points):
    """

    Normalize a collection of points by removing the mean and scaling by the
    variance.

    Returns:
        out : normalized points with mean removed
        mu : the mean
        std : the standard deviation of each coordinate. If all coordinates are
            zero, std is set to 1 to avoid division by zero.

    """
    std = np.std(points,0)
    # Avoid division by zero
    close = np.isclose(std,0*std)
    std[close] = 1
    mu = mean(points)
    out = (points-mu)/std
    return out, mu, std

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

    out = sum([np.tile(points.dot(b[:,i]), 
              (points.shape[1], 1)).T*np.tile(b[:,i], 
              (points.shape[0], 1)) for i in range(b.shape[1])])
    return out



def norms(vecs):
    """
    Return Normalization constants for a collection of vectors:
        
    Arguments:
        vecs : Array of row vectors.  Here, `vecs[0,:]` is the first vector.

    Returns:
        out : Normalization constant for each vector.

    """
    out = np.linalg.norm(vecs, axis=1)
    out = np.tile(out, (vecs.shape[1], 1)).T
    return out

def renormalize(vecs, mu, std):
    """
    Restore normalization and mean value transformations applied to vectors.
    See `normalize` for computation of mean and standard deviation.

    Arguments:
        vecs : Array of row vectors.  Here, `vecs[0,:]` is the first vector.
        mu : Mean value.
        std : Standard deviation.

    """
    return vecs*std + mu

def bbox2(points):
    """
    Compute the bounding box for a set of points in 2D

    Arguments:
        points : An array of size num points x 2 that contains the point cloud.
            Here, `points[0,:]` is the first point with coordinates 
            `x_0, x_1`

    Returns:
        Coordinates that define the bounding box, 
            0: the bottom left,
            1: bottom right, 
            2: top right,
            3: top left coordinate.

    """

    min_v = [0]*2
    max_v = [0]*2
    for i in range(2):
        min_v[i] = np.min(points[:,i])
        max_v[i] = np.max(points[:,i])

    return np.array([[min_v[0], min_v[1]],
                     [max_v[0], min_v[1]], 
                     [max_v[0], max_v[1]],
                     [min_v[0], max_v[1]]]).astype(np.float)

def bbox2_vol(bbox):
    """
    Compute the volume (area) of a two dimensional bounding box

    Arguments:
        bbox : Points defining a bounding box (see bbox2)

    Returns:
        The area of the bounding box.

    """
    return (bbox[1,0] - bbox[0,0])*(bbox[2,1] - bbox[1,1])

def bbox2_expand(bbox, padding):
    """
    Expand the bounding box by applying a padding to each side. The padding is
    defined as a scaling factor of the original size of the bounding box.
    """
    Lx = bbox[1, 0] - bbox[0, 0]
    Ly = bbox[2, 1] - bbox[1, 1]
    Lx, Ly = bbox2_dimensions(bbox)

    out = np.array([[0.0] * 2] * 4)

    # Bottom Left
    out[0, 0] = bbox[0, 0] - padding * Lx
    out[0, 1] = bbox[0, 1] - padding * Ly
    # Bottom right
    out[1, 0] = bbox[1, 0] + padding * Lx
    out[1, 1] = bbox[1, 1] - padding * Ly
    # Top right
    out[2, 0] = bbox[2, 0] + padding * Lx
    out[2, 1] = bbox[2, 1] + padding * Ly
    # Top left
    out[3, 0] = bbox[3, 0] - padding * Lx
    out[3, 1] = bbox[3, 1] + padding * Ly
    return out

def bbox2_dimensions(bbox):
    Lx = bbox[1, 0] - bbox[0, 0]
    Ly = bbox[2, 1] - bbox[0, 1]
    return Lx, Ly



def rotate2(points, mu, theta):
    """
    Perform a 2D coordinate rotation specified by the angular parameter `theta`.
    This parameter increases when the points `xy` are rotated in the counter
    clockwise direction.

    Arguments:
        points : An array of size num points x 2 that contains the point cloud.
            Here, `points[0,:]` is the first point with coordinates 
            `x_0, x_1`
        mu : Point to rotate around.
        theta : Angle to rotate by.

    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    rxy = mu + R.dot(points.T- mu.T).T
    return rxy

def restore(X, Y, Z, basis, mu, std, center, theta):
    """
    Rotate surface back to the original coordinate system.
    Also, remove normalization.

    Args:
        X, Y, Z : x, y, and z-coordinates (can be grid)
        basis : Basis vectors for projection
        mu : Mean
        std : Standard deviation
        center : Center coordinate for 2D in plane rotation
        theta : Signed rotation angle

    Returns:
        X, Y, Z : Restored coordinates.

    """

    xy = np.vstack((X.flatten(), Y.flatten())).T
    center = np.tile(center[0,:], (xy.shape[0],1)) 
    rxy = rotate2(xy, center, -theta)
    xyz = np.vstack((rxy.T, Z.flatten())).T
    xyz = basis.dot(xyz.T).T
    xyz = renormalize(xyz, mu, std)
    if len(X.shape) == 2:
        nx = X.shape[0]
        ny = X.shape[1]
        X = np.reshape(xyz[:,0], (nx, ny))
        Y = np.reshape(xyz[:,1], (nx, ny))
        Z = np.reshape(xyz[:,2], (nx, ny))
    else:
        X = xyz[:,0]
        Y = xyz[:,1]
        Z = xyz[:,2]
    return X, Y, Z

def sdtriangle2(p, p0, p1, p2):
    """
    Signed distance function for a triangle in 2D
    
    Source:
    https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

    Arguments:
        p : Query point.
        p0, p1, p2 : Vertices of triangle.

    Returns:
        out : The signed distance function. `out > 0` if `p` is outside the
        triangle and `out < 0` if `p` is inside the triangle. 

    """
    e0 = p1-p0
    e1 = p2-p1
    e2 = p0-p2
    v0 = p-p0
    v1 = p-p1
    v2 = p-p2

    pq0 = v0 - e0*np.clip( np.dot(v0,e0)/np.dot(e0,e0), 0.0, 1.0 )
    pq1 = v1 - e1*np.clip( np.dot(v1,e1)/np.dot(e1,e1), 0.0, 1.0 )
    pq2 = v2 - e2*np.clip( np.dot(v2,e2)/np.dot(e2,e2), 0.0, 1.0 )
    
    s = np.sign( e0[0]*e2[1] - e0[1]*e2[0] )
    d = min(min((np.dot(pq0,pq0), s*(v0[0]*e0[1]-v0[1]*e0[0])),
                     (np.dot(pq1,pq1), s*(v1[0]*e1[1]-v1[1]*e1[0]))),
                     (np.dot(pq2,pq2), s*(v2[0]*e2[1]-v2[1]*e2[0])))
    out = -np.sqrt(d[0])*np.sign(d[1])
    return out


def sdquad2(p, p0, p1, p2, p3):
    """
    Signed distance function for a quadrilateral element in 2D
    The vertices should be counter clockwise ordered.

    This function is based on computing the signed distance function for a
    triangle by splitting the quad into two triangles and taking the union. A
    side effect of this approach is that the shared edge of the triangles in the
    quad is also accounted for. Hence, this function does not compute the actual
    distance function for a quad.

    Arguments:
        p : Query point.
        p0, p1, p2, p3 : Vertices of quadrilateral.

    Returns:
        out : The signed distance function. `out > 0` if `p` is outside the
        quad and `out < 0` if `p` is inside the quad. 
    """


    d0 = sdtriangle2(p, p0, p1, p3)
    d1 = sdtriangle2(p, p1, p2, p3)
    return min(d0, d1)

def refine(x):
    x1 = 0.5*(x[1::] + x[0:-1])
    x_out = np.zeros((len(x)+len(x1),))
    x_out[::2] = x
    x_out[1:-1:2] = x1
    return x_out


def triangle_vol(p0, p1, p2):
    """
    Compute the area of a triangle. The triangle may be defined in either 2D or
    3D space.

    Arguments:
        p0, p1, p2 : Vertices of the triangle.

    """

    e1 = p1 - p0
    e2 = p2 - p0
    z = np.cross(e1, e2)
    return 0.5*np.linalg.norm(z)

def quad_vol(p0, p1, p2, p3):
    """
    Compute the area of a quadrilateral. The vertices of the quadrilateral must
    be ordered.

    Arguments:
        p0, p1, p2, p3 : Vertices of quadrilateral.
    """
    A1 = triangle_vol(p0, p1, p3)
    A2 = triangle_vol(p1, p2, p3)
    return A1 + A2

def argnearest(points, p, ord=2):
    """
    Find the index of the point that is closest to a query point `p`

    Arguments:
        points : An array of size num points x q that contains the point cloud.
            Here, `points[0,:]` is the first point with coordinates 
            `x_0, x_1, ...`
        p : Query point

    """

    dist = np.linalg.norm(points - np.tile(p, (points.shape[0], 1)), axis=1, 
            ord=ord)
    return np.argmin(dist)

def uv_grid(nu, nv):
    """
    Construct the UV-grid parameterized from 0 to 1.
    """
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)
    return U, V


def bbox2_grid(bbox, nu, nv):
    """
    Construct a grid from a bounding box.

        Args:
            bbox: Bounding box (see bbox2)
            nu: Number of control points in the u-direction
            nv: Number of control points in the v-direction
    """

    # Grid
    Gx = np.linspace(bbox[0, 0], bbox[1, 0], nu)
    Gy = np.linspace(bbox[0, 1], bbox[2, 1], nv)

    X, Y = np.meshgrid(Gx, Gy)
    return X, Y
