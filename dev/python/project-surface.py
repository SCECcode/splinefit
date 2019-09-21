import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage


def main():

    options = get_options(sys.argv)
    sf.options.check_options(sys.argv, options)
    verbose = options.verbose
    savefig = options.savefig
    showfig = options.showfig

    data = pickle.load(open(options.input, 'rb'))

    tris = data.tris

    xyz = rotate(data.pcl_xyz, data.basis, data.proj_basis, 
                 data.theta, data.center)

    bounding_box = sf.fitting.bbox2(xyz)
    bounding_box = sf.fitting.bbox2_expand(sf.fitting.bbox2(xyz),
                                           options.padding)

    bnd_geom = normals(data.tris, data.bnd_edges, xyz)
    bbox_points = intersect(bnd_geom['normals'], bnd_geom['points'],
                            bounding_box, showfig=showfig, savefig=savefig)
    corner_points = set_z_nearest_corners(bbox_points, bounding_box)

    xyz_augmented = np.vstack((xyz, bbox_points, corner_points))


    if options.estimate_uv:
        nu, nv = estimate_uv(xyz, tris, bounding_box, 
                                options.cell_scaling, verbose)
    else:
        nu = options.num_u
        nv = options.num_v

    vprint("Grid dimensions: %d x %d" % (nu, nv), verbose)


    pu = options.degree_u
    pv = options.degree_v

    # Construct uv-grid
    int_knot_u = sf.bspline.numknots(nu, pu, interior=1)
    int_knot_v = sf.bspline.numknots(nv, pv, interior=1)
    U = sf.bspline.uniformknots(int_knot_u, pu)
    V = sf.bspline.uniformknots(int_knot_v, pv)

    # Construct control points
    # Find vertical component of the control points by projecting onto the
    # triangulation
    X, Y = sf.fitting.bbox2_grid(bounding_box, nu, nv)
    queries = np.vstack((X.flatten() , Y.flatten(), 0*X.flatten())).T
    dela, projection = sf.triangulation.project(xyz_augmented, queries)
    Z = np.reshape(projection[:,2], (X.shape[0], Y.shape[1]))
    projection[:,2] = Z.flatten()


    S = sf.bspline.Surface(U, V, pu, pv, X, Y, Z, label='grid')

    if options.fit:
        vprint("Applying surface fitting", verbose)
        res = fit_surface(S, projection, regularization=options.regularization)
        vprint("Residual: %g " % res, verbose)

    # Transform fitted surface to the original coordinate system 
    S.rwPx, S.rwPy, S.rwPz = sf.fitting.restore(S.Px, S.Py, S.Pz,
        data.basis, data.mu, data.std, data.center, data.theta)

    if savefig or showfig:
        S.eval(nu=30, nv=30)
        ax = sf.plot.grid(S.Px, S.Py, S.Pz)
        ax = sf.plot.points3(xyz, 'ko', ax=ax)
        ax.view_init(70, 70)

        if savefig:
            plt.savefig(savefig + "_bspline_surface.png", dpi=300)

        if showfig:
            plt.show()
        plt.close()

    if options.json:
        jsonfile = options.json + ".json"
        S.json(jsonfile)
        vprint("Wrote: %s" % jsonfile, verbose)


def vprint(msg, verbose):
    if not verbose:
        return
    print(msg)

def get_options(argv):
    """
    Get command line arguments.
    """

    options = helper.Struct()
    if '--help' in argv:
        print(helptxt)
        exit()

    args = sf.options.get_options(argv)
    options.input = args['input']
    options.output = args['output']

    if '--degree_u' in args:
        options.degree_u = int(args['--degree_u'])
    else:
        options.degree_u = 3

    if '--degree_v' in args:
        options.degree_v = int(args['--degree_v'])
    else:
        options.degree_v = 3

    if '--num_u' in args:
        options.num_u = int(args['--num_u'])
    else:
        options.num_u = 10

    if '--num_v' in args:
        options.num_v = int(args['--num_v'])
    else:
        options.num_v = 10

    if '--estimate_uv' in args:
        options.estimate_uv = int(args['--estimate_uv'])
    else:
        options.estimate_uv = 1

    if '--padding' in args:
        options.padding = float(args['--padding'])
    else:
        options.padding = 0.1

    if '--cell_scaling' in args:
        options.cell_scaling = float(args['--cell_scaling'])
    else:
        options.cell_scaling = 0.0

    if '--fit' in args:
        options.fit = int(args['--fit'])
    else:
        options.fit = True

    if '--regularization' in args:
        options.regularization = float(args['--regularization'])
    else:
        options.regularization = 1e-1

    if '--json' in args:
        options.json = args['--json']
    else:
        options.json = ''

    if '--verbose' in args:
        options.verbose = int(args['--verbose'])
    else:
        options.verbose = 0
    
    if '--showfig' in args:
        options.showfig = int(args['--showfig'])
    else:
        options.showfig = 0

    if '--savefig' in args:
        options.savefig = args['--savefig']
    else:
        options.savefig = ''

    return options


def rotate(coords, basis, projection_basis, rotation_angle, center):
    """
    Rotate point cloud using the basis vectors of the best fitting plane, and
    bounding box.
    """
    # Rotate point cloud
    xy = projection_basis.T.dot(coords.T).T
    xyz = basis.T.dot(coords.T).T
    center_tiled = np.tile(center[0,:], (xy.shape[0],1)) 
    rxy = sf.fitting.rotate2(xy, center_tiled, rotation_angle)
    xyz[:, 0:2] = rxy
    return xyz

def estimate_uv(points, tris, bbox, cell_scaling, verbose=0):
    """
    Estimate the number of u, v, points to use by determining the average
    element size in the triangulation.

    """
    areas = sf.triangulation.areas(tris, points)

    dist = np.mean(np.sqrt(areas))
    scaled_dist = dist * (1.0 + cell_scaling)
    Lx, Ly = sf.fitting.bbox2_dimensions(bbox)
    vprint("Bounding box dimensions: %d x %d" % (Lx, Ly), verbose)
    vprint("Average distance between points: %g" % dist, verbose) 
    vprint("Scaled distance between points: %g" % scaled_dist, verbose) 

    num_u = round(Lx / scaled_dist ) + 1
    num_v = round(Ly / scaled_dist ) + 1
    return num_u, num_v

def fit_surface(S, points, surf_smooth=0, regularization=0.0):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    u = sf.bspline.xmap(x)
    v = sf.bspline.xmap(y)

    S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv,
                                    s=surf_smooth, a=regularization)
    return res

def normals(tris, bnd_edges, points, showfig=False, savefig=False):
    """
    Find the normals with respect to the boundary triangles. 
    The surface normal is defined using the cross product of two edges of a
    boundary triangle. Hence, this normal is orthogonal to the surface of the
    triangle. The boundary normal is orthogonal to the surface normal and
    tangent vector along the boundary.

    Args:
        tris: An array containing the node indices of each triangle in the mesh
        bnd_edges: An array of the node indices defining the boundary edges.
        points: An array of coordinates for each point in the mesh

    Returns:
        A dictionary containing the surface normals, normals, and boundary
            triangles.
    
    """

    # Extract all edges
    edges = sf.triangulation.tris_to_edges(tris)
    edges_to_nodes = sf.triangulation.edges_to_nodes(edges)

    num_elem = bnd_edges.shape[0]
    normals = np.zeros((num_elem, 3))
    bnd_tris = np.zeros((num_elem, 1))
    surface_normals = np.zeros((num_elem, 3))
    bnd_points = np.zeros((num_elem, 3))

    for k, bnd in enumerate(bnd_edges):
        eds = edges[sf.triangulation.edge_mapping(bnd[0], bnd[1])]
        tri = eds['triangles'][0]
        bnd_tris[k] = tri
        nodes = tris[tri,:]
        pts = points[nodes,:]
        surface_normal = sf.triangulation.normal(pts)
        tangent = points[bnd[1],:] - points[bnd[0],:]
        normal = np.cross(surface_normal, tangent)
        normals[k,:] = normal
        surface_normals[k, :] = surface_normal
        bnd_points[k, :] = points[bnd[0], :]

    if showfig or savefig:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        plt.plot(bnd_points[:,0], bnd_points[:,1], bnd_points[:,2],'k-')
        plt.quiver(bnd_points[:,0], bnd_points[:,1], bnd_points[:,2],
                   normals[:,0], normals[:,1], normals[:,2], length=0.1,
                   normalize=True)
        if savefig:
            plt.savefig(savefig + "normals.png", dpi=300)
            
        if showfig:
            plt.show()

    return {'normals': normals, 
            'tris': bnd_tris,
            'surface_normals': surface_normals,
            'points': bnd_points}

def intersect(normals, points, bbox, showfig=False, savefig=False):
    """
    Compute the intersection point between line defined by the normal, and
    bounding box.
    """

    intersected_points = 0 * normals

    # Bounding box indices
    # 0: bottom left,
    # 1: bottom right, 
    # 2: top right,
    # 3: top left coordinate.

    i = 0
    for ni, pi in zip(normals, points):
        # Upper right quadrant
        xb = bbox[2][0]
        yb = bbox[2][1]
        # Check top
        if ni[1] > 0:
            t = (yb - pi[1]) / ni[1]
            x0 = pi[0] + ni[0] * t
            if (x0 <= xb and x0 >= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [x0, yb, z0]
        # Check right
        if ni[0] > 0:
            t = (xb - pi[0]) / ni[0]
            y0 = pi[1] + ni[1] * t
            if (y0 <= yb and y0 >= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [xb, y0, z0]

        # Upper left quadrant
        xb = bbox[3][0]
        yb = bbox[3][1]
        # Check top
        if ni[1] > 0:
            t = (yb - pi[1]) / ni[1]
            x0 = pi[0] + ni[0] * t
            if (x0 >= xb and x0 <= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [x0, yb, z0]
        # Check left
        if ni[0] < 0:
            t = (xb - pi[0]) / ni[0]
            y0 = pi[1] + ni[1] * t
            if (y0 <= yb and y0 >= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [xb, y0, z0]

        # Bottom right quadrant
        xb = bbox[1][0]
        yb = bbox[1][1]
        # Check bottom
        if ni[1] < 0:
            t = (yb - pi[1]) / ni[1]
            x0 = pi[0] + ni[0] * t
            if (x0 <= xb and x0 >= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [x0, yb, z0]
        # Check right
        if ni[0] > 0:
            t = (xb - pi[0]) / ni[0]
            y0 = pi[1] + ni[1] * t
            if (y0 >= yb and y0 <= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [xb, y0, z0]

        # Bottom left quadrant
        xb = bbox[0][0]
        yb = bbox[0][1]
        # Check bottom
        if ni[1] < 0:
            t = (yb - pi[1]) / ni[1]
            x0 = pi[0] + ni[0] * t
            if (x0 >= xb and x0 <= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [x0, yb, z0]
        # Check left
        if ni[0] < 0:
            t = (xb - pi[0]) / ni[0]
            y0 = pi[1] + ni[1] * t
            if (y0 >= yb and y0 <= 0):
                z0 = pi[2] + ni[2] * t
                intersected_points[i, :] = [xb, y0, z0]

        i += 1

    if showfig or savefig:
        plt.plot(intersected_points[:,0], intersected_points[:,1], 'o')
        plt.plot(points[:,0], points[:,1], 'ko-')
        plt.legend(['bounding box points', 'boundary points'])
        if showfig:
            plt.show()
        if savefig:
            plt.savefig(savefig + "_intersect.png", dpi=300)
        plt.close()
    return intersected_points

def set_z_nearest_corners(points, bounding_box):
    """
    Set the z-value to the points at the bounding box corners to the z-value of
    the nearest points.

    """
    corner_points = np.zeros((4, 3))

    for i in range(4):
        x0 = bounding_box[i][0]
        y0 = bounding_box[i][1]
        dist_idx = np.argmin((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
        corner_points[i,:] = [x0, y0, points[dist_idx, 2]]

    return corner_points




if __name__ == "__main__":
    main()
