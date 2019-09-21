def grid(X, Y, Z=None, color='b', ax=None):
    """
    Plot a structured grid.
    
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if Z is None:
        Z = 0*X
    if not ax:
        fig = plt.figure()
        ax = fig.gca(projection='3d', proj_type = 'ortho')
        ax=fig.gca(projection=Axes3D.name)
    ax.plot_wireframe(X,Y,Z, color=color)
    ax.view_init(90, -90)
    return ax

def points2(points, style='o'):
    import matplotlib.pyplot as plt
    plt.plot(points[:,0], points[:,1], style)

def points3(points, style='o', ax=None):
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

    ax.plot(points[:,0], points[:,1], points[:,2], style)
    return ax
