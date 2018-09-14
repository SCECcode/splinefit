import numpy as np

def close_boundary(points):
    """
    Close the boundary by adding the first point to the end of the array.

    Arguments:
        points : Array of points (size: num points x 3).

    """
    return np.vstack((points, points[0,:]))


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
