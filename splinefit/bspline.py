import numpy as np

def basis(t):
    """
    Evaluate C^2 continuous cubic BSpline basis in each cell i=0,1,2,3.

    Arguments:
        t : Parameter, `0 <= t <= 1`.

    Returns
        out : Value of Bspline basis in each cell.


    """
    B = np.array([[1, -3, 3, -1],
                  [4, 0, -6, 3],
                  [1, 3, 3, -3],
                  [0, 0, 0, 1]])/6
    u = np.array([1, t, t**2, t**3])

    out = B.dot(u)
    return out
