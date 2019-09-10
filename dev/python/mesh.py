from scipy.spatial import Delaunay
import numpy as np

def refine(points, levels=1):

    i = 0
    for level in range(levels):
        new_points = []
        tri = Delaunay(points)
        print("Refinement:", i)
        for t in tri.simplices:
            new_points.append(center_of_gravity(t, tri.points))
        points = np.vstack((tri.points, new_points))
        print(points.shape)
        i += 1
    return tri.points


def project(points, query_points, skip_nan=True, expand=1e-2):
    """
    Project query points in the (x,y)-plane onto a triangulation in (x, y, z).
    This triangulation is determined by the Delaunay triangulation in the (x, y)
    plane.

    Returns:
        tri : Triangulation
        proj : Projection of query points onto the triangulation. Any points
            that fall outside the triangulation are set to Nan unless `skip_nan`
            is True.
        skip_nan : Remove nan-values from output.

    """
    import warnings


    v0 = sum(points[:,0])
    v1 = sum(points[:,1])
    v2 = sum(points[:,2])
    n = points.shape[0]
    cg = [v0/n, v1/n, v2/n]
    distvec = points - np.tile(cg, (points.shape[0], 1))
    points = points + expand*distvec

    deltri = Delaunay(points[:,0:2])
    tris = deltri.find_simplex(query_points[:,0:2])

    proj = np.zeros((query_points.shape[0], 3))
    tol = 1e-12
    for i, tri in enumerate(tris):
        # Skip query points outside domain
        if tri == -1:
            proj[i,:] = 0#np.nan
            continue

        tripoints = points[deltri.simplices[tri,:],:]
        n = normal(tripoints)
        # Determine z-coordinate for query point (x,y), where z lies on the
        # triangle `tri`
        qp = query_points[i,:]
        tp = tripoints[0,:]
        proj[i,0:2] = qp[0:2]
        if abs(n[2]) < tol:
            warnings.warn('Triangle is near orthogonal to projection plane')

        proj[i,2] = -(n[0]*(qp[0] - tp[0]) + n[1]*(qp[1] - tp[1]))/n[2] + tp[2]

    if skip_nan:
        proj = proj[~np.isnan(proj[:,0]),:]

    return deltri.simplices.copy(), proj

def normal(points):
    """

    Compute the outward facing normal with respect to a triangle

    """
    
    u = points[2,:] - points[1,:]
    v = points[0,:] - points[1,:]
    return np.cross(u, v)
    



def center_of_gravity(t, points):
    p0 = points[t[0],:]
    p1 = points[t[1],:]
    p2 = points[t[2],:]
    return (p0 + p1 + p2)/3

