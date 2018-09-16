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
    return int(np.floor(pts) - 1)

def upper(pts):
    """
    Same as `lower`, but returns the highest index.
    """
    return int(np.floor(pts) + 2)

def eval(P, npts=10):

    t = np.linspace(0, 1, npts)
    x = np.zeros(((len(P)-3)*npts,))
    y = np.zeros(((len(P)-3)*npts,))
    k = 0
    for i in range(1,len(P)-2):
        ctrlpts = P[i-1:i+3]
        for ti in t:
            qx = cubic(ti).dot(ctrlpts)
            y[k] = qx
            x[k] = i + ti - 1
            k += 1
    return x, y

def eval_basis(xc, n):
    idx = np.floor(xc)
    t = (xc - idx)/n
    w = cubic(t)
    return w

def set_ctrlpt(xc, yc, n):
    """

    Assign value to control point by solving the constrained optimization
    problem:

    min || \sum_a P_a ||^2 subject to \sum_a w_a P_a = yc

    Solution is given by:

    P_k = w_k*yc/(sum_a w_a^2)

    Arguments:
        xc : x-coordinate of point. Determines which control point to
            influence. 
        yc : y-coordinate of point. Determines what weight to assign to the
            affected control point.
         n : Number of control points - 1

    Returns:
        val : Value of control point.
        idx : Index of control point.

    """

    w = eval_basis(xc, n)
    return w*yc/(w.dot(w)), (lower(xc) + 2, upper(xc) + 3) 


def findspan(n, p, u, U):
    """

    n : Number of control points
    p : degree
    u : parameter to find span for, should lie in 0 <= u <= 1.
    U : knot vector

    """
    if u == U[n+1]:
        return n

    for i in range(p, p + n):
        if u >= U[i] and u < U[i+1]:
            return i
    return n
    low = p
    high = n + 1
    mid = int((low+high)/2)
    while (u < U[mid] or u >= U[mid+1]):
        if (u < U[mid]):
            high = mid
        else:
            low = mid
        mid = int((low + high)/2)
    return mid



def basisfuns(i,u,p,U):
    N = np.zeros((p+1,))
    left = np.zeros((p+1,))
    right = np.zeros((p+1,))
    N[0] = 1.0
    for j in range(1,p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r]/(right[r+1] + left[j-r])
            N[r] = saved+right[r+1]*temp
            saved = left[j-r]*temp
        N[j] = saved

    return N

def curvepoint(p, U, P, u):
    C = 0.0
    n = len(P) - p
    span = int(np.floor(u)) + p
    span = findspan(n, p, u, U)
    N = basisfuns(span,u,p,U)
    for i in range(p+1):
        C = C + N[i]*P[span-p+i]
    return C

def bspline_lsq(x, y, U, p):
    """
    Computes the least square fit to the data (x,y) using the knot vector U.

    Arguments:
        x, y : Data points
        U : Knot vector
        p : Degree of BSpline

    Returns:
        Control points P

    """
    assert len(x) == len(y)
    nctrl = len(U) - 1
    n = nctrl - p - 1
    npts = len(x)
    P = np.zeros((nctrl,))

    A = np.zeros((npts, nctrl))
    b = np.zeros((npts,))
    for i, xi in enumerate(x):
        span = findspan(n, p, xi, U)
        N = basisfuns(span, xi, p, U)
        for j, Nj in enumerate(N):
            A[i, span + j - p] = Nj
        b[i] = y[i]

    p0 = np.linalg.lstsq(A, b, rcond=None)[0]
    P[0:nctrl] = p0
    return P


