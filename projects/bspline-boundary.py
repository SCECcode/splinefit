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

if len(sys.argv) < 6:
    figfile = None
else:
    figfile = sys.argv[5]

def make_curves(data1, data2, p, sm, disp=True):
    curve1 = helper.Struct()
    curve1.x = data1[:,0]
    curve1.y = data1[:,1]
    curve2 = helper.Struct()
    curve2.x = data2[:,0]
    curve2.y = data2[:,1]
    curve1.Px, curve1.Py, curve1.U, curve2.Px, curve2.Py, curve2.U =\
            fit(curve1.x, curve1.y, curve2.x, curve2.y, p, sm)
    return curve1, curve2

def get_num_ctrlpts(bnd):
    return bnd.Px.shape[0] - p

def fit(x1, y1, x2, y2, p, sm, disp=True, mmax=100):
    """
    Perform least squares fitting by successively increasing the number of knots
    until a desired residual threshold is reached.

    Returns:
        Px, Py : Control points
        U : Knot vector

    """
    m = 1
    it = 0
    res = sm + 1
    while (res > sm and m < mmax):
        it += 1
        Px1, Py1, U1, res1 = sf.bspline.lsq2l2(x1, y1, m, p)
        Px2, Py2, U2, res2 = sf.bspline.lsq2l2(x2, y2, m, p)
        res = np.linalg.norm(res1) + np.linalg.norm(res2)
        if disp:
            print("Iteration: %d, number of interior knots: %d, residual: %g" %
                    (it, m, res))
        m = 2+m
    return Px1, Py1, U1, Px2, Py2, U2

def make_plot(curve, figfile, save=0, npts=100):
    if not figfile:
        return
    u = np.linspace(curve.U[0], curve.U[-1], 100)
    cx = sf.bspline.evalcurve(p, curve.U, curve.Px, u)
    cy = sf.bspline.evalcurve(p, curve.U, curve.Py, u)
    plt.plot(curve.x,curve.y,'ko')
    plt.plot(cx,cy,'-')

    if save:
        plt.savefig(figfile)

data = pickle.load(open(inputfile, 'rb'))

print("Determining number of u knots..")
bottom, top = make_curves(data.bottom, data.top, p, sm, disp=1)
print("Determining number of v knots..")
left, right = make_curves(data.left, data.right, p, sm, disp=1)

mu = max(get_num_ctrlpts(left), get_num_ctrlpts(right))
mv = max(get_num_ctrlpts(top), get_num_ctrlpts(bottom))

print("Number of UV control points: [%d, %d]" % (mu, mv))

make_plot(left, figfile)
make_plot(bottom, figfile)
make_plot(right, figfile)
make_plot(top, figfile, save=1)
plt.show()
