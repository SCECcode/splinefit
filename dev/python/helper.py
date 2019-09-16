import numpy as np
import splinefit as sf

def get_args(args, delimiter="="):
    d_args = {}
    for arg in args[1::]:
        key, value = arg.split(delimiter)
        d_args[key] = value
    return d_args

def close_boundary(points):
    """
    Close the boundary by adding the first point to the end of the array.

    Arguments:
        points : Array of points (size: num points x 3).

    """
    return np.vstack((points, points[0,:]))

def evalcurve(curve, num=100):
    u = np.linspace(curve.U[0], curve.U[-1], num)
    cx = sf.bspline.evalcurve(curve.p, curve.U, curve.Px, u)
    cy = sf.bspline.evalcurve(curve.p, curve.U, curve.Py, u)
    return cx, cy

def evalcurve3(curve, num):
    u = np.linspace(curve.U[0], curve.U[-1], num)
    cx = sf.bspline.evalcurve(curve.p, curve.U, curve.Px, u)
    cy = sf.bspline.evalcurve(curve.p, curve.U, curve.Py, u)
    cz = sf.bspline.evalcurve(curve.p, curve.U, curve.Pz, u)
    return cx, cy, cz

def plot_curve(x, y, z, ax, style='C1o-'):
    pts = np.vstack((x, y, z)).T
    plot_points(pts, ax=ax, style=style)

def plot_mesh(points, triangles, ax=None):
    """
    Plot triangular mesh.
    
    Arguments:
        points : Array of points (size: num points x 3).
        triangles : Array of triangular elements (size: num triangles x 3).
        ax (optional) : Pass if already exists.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=triangles,
                    shade=False)
    return fig, ax

def plot_grid(X, Y, Z=0, ax=None, color='b'):
    """
    Plot a structured grid.
    
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d', proj_type = 'ortho')
        ax=fig.gca(projection=Axes3D.name)
    ax.plot_wireframe(X,Y,Z, color=color)
    ax.view_init(90, -90)
    return ax

def plot_points(points, ax=None, style='-'):
    """
    Plot points in 3D space.

    Arguments:
        points : Array of points (size: num points x 3).
        style : plot style.
        ax (optional) : Pass if already exists.

    Returns:
        fig : figure handle.
        ax : Axis handle.

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    ax.plot(points[:,0], points[:,1], points[:,2],style)
    return ax

def plot_points2(points, style='-'):
    import matplotlib.pyplot as plt

    plt.plot(points[:,0], points[:,1],style)


def show(show_plot):
    import matplotlib.pyplot as plt
    if show_plot:
        plt.show()


def plot_basis(basis, ax=None, style=''):
    """
    Plot basis vectors in 3D space

    Arguments:
        basis : Basis vectors (size: dimension x num basis).
        ax (optional) : Pass if already exists.
        style : plot style.

    Returns:
        fig : figure handle
        ax : Axis handle

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    vec = lambda x, comp, basis : [0., x[comp,basis]]
    for basis_j in range(basis.shape[1]):
        ax.plot(*[vec(basis, i, basis_j) for i in range(basis.shape[0])], style)

    return ax

class Struct(dict):
    """
    Make a dict behave as a struct.

    Example:
    
        test = Struct(a=1, b=2, c=3)

    """
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

def export_msh(coords, bnd_edges, outputfile):
    #TODO: Export to gmsh. This exporter is not yet working.
    num_edges = bnd_edges.shape[0]
    # id, elem type, num tags, tags, node list
    num_attr = 6
    elems = np.zeros((num_edges, num_attr)).astype(np.int64)
    for i in range(num_edges):
        elems[i,0] = i + 1
        elems[i,1] = 1
        elems[i,2] = 1
        elems[i,3] = 1
        elems[i,4] = bnd_edges[i,0] + 1
        elems[i,5] = bnd_edges[i,1] + 1
    sf.msh.write(outputfile, coords, elems)
