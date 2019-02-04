import splinefit as sf
from splinefit import msh
import numpy as np
import helper
import matplotlib.pyplot as plt

def make_curves(data1, data2, p, sm, s, a=0, disp=True, axis=0, mmax=40, ratio=3):
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]

    x1, y1, z1, x2, y2, z2 = refine_curves(x1, y1, z1,
                                           x2, y2, z2,
                                           p,
                                           ratio=ratio)

    mmax = min_degree(x1, x2, p, mmax=mmax)
    

    curve1, curve2 = fit(x1, y1, z1, x2, y2, z2, p, sm, s, 
                         disp=disp, axis=axis, mmax=mmax, a=a)
    curve1.px = curve1.Px[:-p]
    curve1.py = curve1.Py[:-p]
    curve2.px = curve2.Px[:-p]
    curve2.py = curve2.Py[:-p]
    return curve1, curve2

def make_curve(data, p, sm, s, disp=True, a=0, axis=0, mmax=40, ratio=3):
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    mmax = min_degree(x, x, p, mmax=mmax)

    m = len(x)
    curve, res = fit_curve(x, y, z, p, m, s, a=a)
    curve.px = curve.Px[:-p]
    curve.py = curve.Py[:-p]
    return curve

def get_num_ctrlpts(bnd):
    return bnd.Px.shape[0]


def refine_curves(x1, y1, z1, x2, y2, z2, p, ratio=3.0):
    # Check that number of points on each side is balanced, otherwise refine

    is_balanced = False

    while not is_balanced:
        len1 = len(x1)
        len2 = len(x2)

        if len1 - p < 1:
            x1 = sf.fitting.refine(x1)
            y1 = sf.fitting.refine(y1)
            z1 = sf.fitting.refine(z1)
            print("Refine 1 (too few points)")
        
        if len2 - p < 1:
            x2 = sf.fitting.refine(x2)
            y2 = sf.fitting.refine(y2)
            z2 = sf.fitting.refine(z2)
            print("Refine 2 (too few points)")

        # Refine 2
        if len1 > len2 and len1/len2 > ratio:
            x2 = sf.fitting.refine(x2)
            y2 = sf.fitting.refine(y2)
            z2 = sf.fitting.refine(z2)
            print("Refine 2")
        # Refine 1
        elif len1 < len2 and len2/len1 > ratio:
            x1 = sf.fitting.refine(x1)
            y1 = sf.fitting.refine(y1)
            z1 = sf.fitting.refine(z1)
            print("Refine 1")
        else:
            is_balanced = True
    return x1, y1, z1, x2, y2, z2

def min_degree(x1, x2, p, mmax=40):
    out = max(min(mmax, len(x1) - p , len(x2) - p), 1)
    return out

def interior_knots(p):
    # Interior knots. No less than p + 1 control points
    m = (p + 1) - 2
    return m


def fit(x1, y1, z1,  x2, y2, z2, p, sm, s, a=0, disp=False, mmax=40, axis=0):
    """
    Perform least squares fitting by successively increasing the number of knots
    until a desired residual threshold is reached.

    Returns:
        Px, Py : Control points
        U : Knot vector
        int_knot : Number of interior knots
        p : Degree
        sm : Desired residual
        s : Smoothing
        a : Second derivative regularization parameter


    """
    import warnings
    m = interior_knots(p)
    it = 0
    res = sm + 1
    mmax = min_degree(x1, x2, p, mmax=mmax)

    if mmax <= m:
        warnings.warn("Too few data points for given polynomial degree.\
                       Refining dataset using averages.")

        x2 = sf.fitting.refine(x2)
        y2 = sf.fitting.refine(y2)
        z2 = sf.fitting.refine(z2)
        p = mmax 
        m = mmax - 1
        print("New polynomial degree:", p)

    best_res = 1e6
    while (res > sm and m < mmax):
        it += 1

        Px1, Py1, U1, res1 = sf.bspline.lsq2l2(x1, y1, m, p, smooth=s)
        Px2, Py2, U2, res2 = sf.bspline.lsq2l2(x2, y2, m, p, smooth=s)
        s1 = sf.bspline.chords(x1, y1)
        s2 = sf.bspline.chords(x2, y2)
        Pz1, rz = sf.bspline.lsq(s1, z1, U1, p, s=s)
        Pz2, rz = sf.bspline.lsq(s2, z2, U2, p, s=s)
        curve1, res1 = fit_curve(x1, y1, z1, p, m, s, a=a)
        curve2, res2 = fit_curve(x2, y2, z2, p, m, s, a=a)

        # Monitor differences
        dPx1 = max(abs(curve1.Px[1:] - curve1.Px[0:-1]))
        dPx2 = max(abs(curve2.Px[1:] - curve2.Px[0:-1]))
        dPy1 = max(abs(curve1.Py[1:] - curve1.Py[0:-1]))
        dPy2 = max(abs(curve2.Py[1:] - curve2.Py[0:-1]))
        res = np.linalg.norm(res1) + np.linalg.norm(res2)
        res = res1 + res2
        res = dPx1 + dPx2 + dPy1 + dPy2
        if res < best_res:
            best_iter = it
            best_res = res
            best_m = m
        if disp:
            print("Iteration: %d, number of interior knots: %d, residual: %g" %
                    (it, m, res))
            n = len(Px1)
        m = 2+m
        int_knot = m - 2

        curve1, res = fit_curve(x1, y1, z1, p, m, s, a=a)
        curve2, res = fit_curve(x2, y2, z2, p, m, s, a=a)

    return curve1, curve2


def fit_curve(x, y, z, p, m, s, a=0.5, tol=1e-6):
    """
    Fit BSpline curve using linear least square approximation with second
    derivative regularization.
    """

    xm = np.mean(x)
    ym = np.mean(y)
    zm = np.mean(z)
    t = sf.bspline.chords(x-xm, y-ym)
    U = sf.bspline.uniformknots(m, p)
    l = np.zeros((len(U) - p - 1,))
    wx = 1.0 + 0 * l
    wy = 1.0 + 0 * l
    wz = 1.0 + 0 * l
    Px, rx = sf.bspline.lsq(t, x - xm, U, p, tol=tol, s=0, a=a, w=wx)
    Py, ry = sf.bspline.lsq(t, y - ym, U, p, tol=tol, s=0, a=a, w=wy)
    Pz, rz = sf.bspline.lsq(t, z - zm, U, p, tol=tol, s=0, a=a, w=wz)

    curve = helper.Struct()
    curve.x = x
    curve.y = y
    curve.z = z
    curve.Px = Px + xm
    curve.Py = Py + ym
    curve.Pz = Pz + zm
    curve.U = U
    curve.p = p
    curve.u = t
    curve.int_knot = m

    res = rx + ry + rz
    return curve, res

def fit_curve_2(x, y, z, p, sm, s, disp=False, mmax=40, axis=0):
    """
    Perform least squares fitting by successively increasing the number of knots
    until a desired residual threshold is reached.

    This function operates on only one curve.

    Returns:
        Px, Py : Control points
        U : Knot vector
        int_knot : Number of interior knots
        p : Degree
        sm : Desired residual
        s : Smoothing


    """
    import warnings
    import scipy.interpolate
    m = interior_knots(p)
    it = 0
    res = sm + 1
    mmax = min_degree(x, x, p, mmax=mmax)

    if mmax <= m:
        warnings.warn("Too few data points for given polynomial degree.\
                       Refining dataset using averages.")

        x = sf.fitting.refine(x)
        y = sf.fitting.refine(y)
        z = sf.fitting.refine(z)
        p = mmax 
        m = mmax - 1
        print("New polynomial degree:", p)

    best_res = 1e6
    eps = 1e-5
    while (res > sm and m < mmax):
        it += 1
        xm = np.mean(x)
        ym = np.mean(y)
        zm = np.mean(z)
        t = sf.bspline.chords(x-xm, y-ym)
        U = sf.bspline.uniformknots(m, p)

        wx = 0 * U + 1.0
        wy = 0 * U + 1.0
        old_res = res + 1
        while (res < old_res):
            Px, rx = sf.bspline.lsq(t, x - xm, U, p, tol=1e-1, s=0, a=0.0,  w=wx)
            Py, ry = sf.bspline.lsq(t, y - ym, U, p, tol=1e-1, s=0, a=0.0, w=wy)
        #Px, Py, res = sf.bspline.lsq2(s1, x-xm, y-ym, U, p, smooth=0)
            old_res = res - 1
            res = rx + ry
            wx = 1.0/(eps**2 + (Px[1:] - Px[0:-1])**2)
            wy = 1.0/(eps**2 + (Py[1:] - Py[0:-1])**2)
            print("wx", wx)
            print("wy", wy)
            print("Residual:", res, sm)
        Pz, rz = sf.bspline.lsq(t, z-zm, U, p)
        Px += xm
        Py += ym
        Pz += zm

        # Monitor differences
        dPx = max(abs(Px[1:] - Px[0:-1]))
        dPy = max(abs(Py[1:] - Py[0:-1]))
        #res = np.linalg.norm(res)
        res = dPx + dPy
        print(res, sm, m, mmax)
        if res < best_res:
            best_iter = it
            best_res = res
            best_m = m
        if disp:
            print("Iteration: %d, number of interior knots: %d, residual: %g" %
                    (it, m, res))
            n = len(Px)
        m = 2+m
        int_knot = m - 2

        curve = helper.Struct()
        curve.x = x
        curve.y = y
        curve.z = z
        curve.Px = Px
        curve.Py = Py
        curve.Pz = Pz
        curve.U = U
        curve.p = p
        curve.u = t
        curve.int_knot = int_knot
        #make_plot(curve,'ff')
        #plt.show()
    return curve

from mpl_toolkits.mplot3d import Axes3D
fig= plt.figure()
ax = fig.gca(projection='3d')

def make_plot(curve, figfile, save=0, npts=100, color=1, p3d=1):
    if not figfile:
        return
    cx, cy, cz = helper.evalcurve3(curve, npts)

    if p3d:
        print(curve.z)
        curve_points = np.vstack((cx, cy, cz)).T
        points = np.vstack((curve.x, curve.y, curve.z)).T
        helper.plot_points(curve_points, ax=ax, style='-')
        helper.plot_points(points, ax=ax, style='k-o')
    else:
        plt.plot(curve.x,curve.y,'k-o')
        plt.plot(cx,cy,'C%d-'%color)
        plt.plot(curve.Px, curve.Py, 'C%do-'%color, alpha=0.3)


    if save:
        plt.savefig(figfile)
