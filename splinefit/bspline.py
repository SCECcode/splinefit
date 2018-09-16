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
