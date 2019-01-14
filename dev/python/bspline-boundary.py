import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
outputfile = sys.argv[2]
p = int(sys.argv[3])
sm = float(sys.argv[4])
s = float(sys.argv[5])
refine_ratio = float(sys.argv[6])

if len(sys.argv) < 8:
    figfile = None
else:
    figfile = sys.argv[7]

if len(sys.argv) < 9:
    showplot = 0
else:
    showplot = int(sys.argv[8])

def make_curves(data1, data2, p, sm, s, disp=True, axis=0, mmax=40):
    x1 = data1[:,0]
    y1 = data1[:,1]
    z1 = data1[:,2]
    x2 = data2[:,0]
    y2 = data2[:,1]
    z2 = data2[:,2]

    x1, y1, z1, x2, y2, z2 = refine_curves(x1, y1, z1,
                                           x2, y2, z2,
                                           ratio=refine_ratio)

    mmax = min_degree(x1, x2, p, mmax=mmax)
    

    curve1, curve2 = fit(x1, y1, z1, x2, y2, z2, p, sm, s, 
                         disp=disp, axis=axis, mmax=mmax)
    curve1.px = curve1.Px[:-p]
    curve1.py = curve1.Py[:-p]
    curve2.px = curve2.Px[:-p]
    curve2.py = curve2.Py[:-p]
    return curve1, curve2

def get_num_ctrlpts(bnd):
    return bnd.Px.shape[0]


def refine_curves(x1, y1, z1, x2, y2, z2, ratio=3.0):
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


def fit(x1, y1, z1,  x2, y2, z2, p, sm, s, disp=False, mmax=40, axis=0):
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

        # Monitor differences
        dPx1 = max(abs(Px1[1:] - Px1[0:-1]))
        dPx2 = max(abs(Px2[1:] - Px2[0:-1]))
        dPy1 = max(abs(Py1[1:] - Py1[0:-1]))
        dPy2 = max(abs(Py2[1:] - Py2[0:-1]))
        #res0 = res
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

        Px1, Py1, U1, res1 = sf.bspline.lsq2l2(x1, y1, best_m, p, smooth=s)
        Px2, Py2, U2, res2 = sf.bspline.lsq2l2(x2, y2, best_m, p, smooth=s)
        s1 = sf.bspline.chords(x1, y1)
        s2 = sf.bspline.chords(x2, y2)
        Pz1, rz = sf.bspline.lsq(s1, z1, U1, p, s=s)
        Pz2, rz = sf.bspline.lsq(s2, z2, U2, p, s=s)

        curve1 = helper.Struct()
        curve1.x = x1
        curve1.y = y1
        curve1.z = z1
        curve1.Px = Px1
        curve1.Py = Py1
        curve1.Pz = Pz1
        curve1.U = U1
        curve1.p = p
        curve1.u = s1
        curve1.int_knot = int_knot
        curve2 = helper.Struct()
        curve2.x = x2
        curve2.y = y2
        curve2.z = z2
        curve2.Px = Px2
        curve2.Py = Py2
        curve2.Pz = Pz2
        curve2.U = U2
        curve2.p = p
        curve2.u = s2
        curve2.int_knot = int_knot
    return curve1, curve2

def make_plot(curve, figfile, save=0, npts=100, color=1):
    if not figfile:
        return
    cx, cy, cz = helper.evalcurve3(curve, npts)
    plt.plot(curve.x,curve.y,'k-o')
    plt.plot(cx,cy,'C%d-'%color)
    plt.plot(curve.Px, curve.Py, 'C%do-'%color, alpha=0.3)

    if save:
        plt.savefig(figfile)
data = pickle.load(open(inputfile, 'rb'))

print("Determining number of u-knots...")
bottom, top = make_curves(data.bottom, data.top, p, sm, s, disp=True, axis=0)
print("Determining number of v-knots...")
left, right = make_curves(data.left, data.right, p, sm, s, disp=True, axis=1)

mu = max(get_num_ctrlpts(left), get_num_ctrlpts(right))
mv = max(get_num_ctrlpts(top), get_num_ctrlpts(bottom))

print("Number of UV control points: [%d, %d]" % (mu, mv))

make_plot(left, figfile, color=0)
make_plot(bottom, figfile, color=1)
make_plot(right, figfile, color=2)
make_plot(top, figfile, color=3, save=1)
helper.show(showplot)


pickle.dump((left, right, bottom, top, data), open(outputfile, 'wb'))
