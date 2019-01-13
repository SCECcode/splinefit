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

def make_curves(data1, data2, p, sm, s, disp=True, axis=0):
    curve1 = helper.Struct()
    curve1.x = data1[:,0]
    curve1.y = data1[:,1]
    curve2 = helper.Struct()
    curve2.x = data2[:,0]
    curve2.y = data2[:,1]

    curve1.x, curve1.y, curve2.x, curve2.y = refine_curves(curve1.x, curve1.y,
                                                           curve2.x, curve2.y,
                                                           ratio=refine_ratio)

    curve1.Px, curve1.Py, curve1.U, curve2.Px, curve2.Py, curve2.U, p, int_knot\
            = fit(curve1.x, curve1.y, curve2.x, curve2.y, p, sm, s, 
                  disp=disp, axis=axis)
    curve1.p = p
    curve2.p = p
    curve1.int_knot = int_knot
    curve2.int_knot = int_knot
    curve1.px = curve1.Px[:-p]
    curve1.py = curve1.Py[:-p]
    curve2.px = curve2.Px[:-p]
    curve2.py = curve2.Py[:-p]
    return curve1, curve2

def get_num_ctrlpts(bnd):
    return bnd.Px.shape[0]


def refine_curves(x1, y1, x2, y2, ratio=3.0):
    # Check that number of points on each side is balanced, otherwise refine

    is_balanced = False

    while not is_balanced:
        len1 = len(x1)
        len2 = len(x2)

        # Refine 2
        if len1 > len2 and len1/len2 > ratio:
            x2 = sf.fitting.refine(x2)
            y2 = sf.fitting.refine(y2)
            print("Refine 2")
        # Refine 1
        elif len1 < len2 and len2/len1 > ratio:
            x1 = sf.fitting.refine(x1)
            y1 = sf.fitting.refine(y1)
            print("Refine 1")
        else:
            is_balanced = True
    return x1, y1, x2, y2

def fit(x1, y1, x2, y2, p, sm, s, disp=False, mmax=40, axis=0):
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
    # Interior knots. No less than p + 1 control points
    m = (p + 1) - 2
    it = 0
    res = sm + 1
    mmax = max(min(mmax, len(x1) - p , len(x2) - p), 1)
    

    if mmax <= m:
        warnings.warn("Too few data points for given polynomial degree.\
                       Reducing polynomial approximation")
        p = mmax 
        m = mmax - 1
        print("New polynomial degree:", p)



    while (res > sm and m < mmax):
        it += 1
        Px1, Py1, U1, res1 = sf.bspline.lsq2l2(x1, y1, m, p, smooth=s)
        Px2, Py2, U2, res2 = sf.bspline.lsq2l2(x2, y2, m, p, smooth=s)
        res = np.linalg.norm(res1) + np.linalg.norm(res2)
        if disp:
            print("Iteration: %d, number of interior knots: %d, residual: %g" %
                    (it, m, res))
        m = 2+m
        int_knot = m - 2
    return Px1, Py1, U1, Px2, Py2, U2, p, int_knot

def make_plot(curve, figfile, save=0, npts=100, color=1):
    if not figfile:
        return
    cx, cy = helper.evalcurve(curve, npts)
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
