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

    header = \
    "# vtk DataFile Version 3.0\n"\
    + "vtk output\n"\
    + "ASCII\n"\
    + "DATASET UNSTRUCTURED_GRID\n"
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
