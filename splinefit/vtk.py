import numpy as np
def write_triangular_mesh(filename, p, t):
    """
    Write a triangular mesh using the VTK legacy file format.

    Input arguments:
        filename : Name of file to write to.
        p : Array of points,  size : (num points, 3) (columns: x, y, z)
        t : Array of element connectivity, size : (num elements, 3)
    """

    npts = p.shape[0]
    dim = p.shape[1]
    ntris = t.shape[0]
    assert dim == 3

    f = open(filename, 'w')

    header = write_header() + "DATASET UNSTRUCTURED_GRID\n"
    points = "POINTS %d float\n" % npts
    cells = "CELLS %d %d\n" % (ntris, 4*ntris)
    cell_types = "CELL_TYPES %d\n" % (ntris)
    cell_type = 5
    num_nodes = 3


    f.write(header)
    f.write(points)
    # x, y, z coordinates
    for i in range(npts):
        f.write('%g %g %g\n' % (p[i,0], p[i,1], p[i,2]))

    f.write(cells)
    # Cells contain the number of nodes, followed by the nodes
    for i in range(ntris):
        f.write('%d %d %d %d\n' % (num_nodes, t[i,0], t[i,1], t[i,2]))

    f.write(cell_types)
    for i in range(ntris):
        f.write('%d\n'% cell_type)

    f.close()

def write_header():
    header = "# vtk DataFile Version 3.0\n"\
    + "vtk output\n"\
    + "ASCII\n"
    return header

def write_surface(filename, X, Y, Z):
    """
    Write a Surface in 3D as structured grid in the VTK legacy file format.

    Input arguments:
        filename : Name of file to write to.
        X : meshgrid of points in the x-direction 
        Y : meshgrid of points in the y-direction 
        Z : meshgrid of points in the z-direction 
    """
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == Z.shape[0]
    assert X.shape[1] == Y.shape[1]
    assert X.shape[1] == Z.shape[1]


    f = open(filename, 'w')

    header = write_header() + "DATASET STRUCTURED_GRID\n"
    dim = "DIMENSIONS %d %d %d\n" %(X.shape[0], X.shape[1], 1)
    points = "POINTS %d float\n" % (X.shape[0]*X.shape[1])

    f.write(header)
    f.write(dim)
    f.write(points)
    # x, y, z coordinates
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            f.write('%g %g %g\n' % (X[i,j], Y[i,j], Z[i,j]))
    f.close()
