import numpy as np

def cubic(t):
    """
    Evaluate C^2 continuous cubic BSpline basis for each function B0, B1, B2, B3.

    out = [t^3, t^2, t, 1] * B

    To compute the Bspline curve, simply dot the resulting values with the
    control points.

    C(u) = [t^3, t^2, t, 1] * B_i * P, P = [P_i, P_{i+1}, P_{i+2}, P_{i+3}],

    where P are the control points.

    Arguments:
        t : Parameter, `0 <= t <= 1`.

    Returns
        out : Value of Bspline basis in each cell.


    """
    B= np.array([[-1, 3, -3, 1],
                 [3, -6, 3, 0],
                 [-3, 0, 3, 0],
                 [1, 4, 1, 0]])/6
    u = np.array([t**3, t**2, t, 1])

    out = u.dot(B)
    return out

def normalize(pts, n):
    """
    Renormalize points so that for each point `pt`, we have `0 <= pt <= n`.

    Arguments:
        pts : List of points (array of size (m,))
        n : Number of control points - 1.

    Returns:
        out : Normalized points
        nc : Normalization constants

    """
    pmin = np.min(pts) 
    pmax = np.max(pts)
    out = n*(pts - pmin)/(pmax - pmin)
    nc = (pmin, pmax)
    return out, nc 

def denormalize(npts, nc, n):
    """
    Restore normalized points, to their original, unnormalized representation

    Arguments:
        npts : List of normalized points (array of size (m,))
        nc : Normalization constants (see `normalize`)
        n : Number of control points - 1.

    Returns:
        out : Normalized points

    """
    out = npts*(nc[1] - nc[0])/n + nc[0]
    return out

def lower(pts):
    """
    Determine the lowest index of the control points that lie in the
    neighborhood of the query points `pts`.

    Arguments:
        pts : Query points, must be normalized to 0 <= pts <= n, where n+1 is
            the number of control points.
    """
    return np.floor(pts) - 1

def upper(pts):
    """
    Same as `lower`, but returns the highest index.
    """
    return np.floor(pts) + 2
