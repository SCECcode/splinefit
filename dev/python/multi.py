from mba import *
import matplotlib.pyplot as plt
import mba
import numpy as np
import pickle
import splinefit as sf
import helper


def fit(m0, level, x, y, p=3):

    curve = helper.Struct()
    mu = sf.multilevel.refine(m0, level)
    curve.U = sf.bspline.uniformknots(mu, p)
    curve.p = p

    curve.s = x#sf.bspline.chords(x, y, a=0, b=1)
    curve.Px, res = sf.bspline.lsq(curve.s, x, curve.U, p)
    curve.Py, res = sf.bspline.lsq(curve.s, y, curve.U, p)
    return curve

def evalcurve(curve):
    cx = sf.bspline.evalcurve(curve.p, curve.U, curve.Px, curve.s)
    cy = sf.bspline.evalcurve(curve.p, curve.U, curve.Py, curve.s)
    return cx, cy



pu = 3
m0 = 3
mu0 = sf.multilevel.refine(m0, 0)
mu1 = sf.multilevel.refine(m0, 1)
levels=1
idx0 = sf.multilevel.indices(0, m0, levels)
idx1 = sf.multilevel.indices(1, m0, levels)
kidx0 = sf.multilevel.knot_indices(idx0, pu)
kidx1 = sf.multilevel.knot_indices(idx1, pu)
U0 = sf.bspline.uniformknots(mu0, pu)
U1 = sf.bspline.uniformknots(mu1, pu)
npts = 100
x = np.linspace(0, 1, npts)
y = np.sin(10*x)

levels=2
master = fit(m0, levels-1, x, y)
print(master.Px.shape)
for i in range(levels):
    curve = fit(m0, i, x, y)
    cx, cy = helper.evalcurve(curve, 100)
    dx, dy = evalcurve(curve)
    idx0 = sf.multilevel.ctrl_indices(i, m0, levels)
    #print(idx0)
    print("curve", curve.Px)
    #print("master",master.Px[idx0])
    #print(curve.Px.shape)
    #plt.plot(x, y)
    #plt.plot(cx, cy)
    #plt.plot(x, dy-y)
#plt.show()
exit(1)

# Length of control points is len(U) - p
nctrl1 = len(U1) - pu
P = range(nctrl1)
print(P)

exit(1)


print(list(range(mu1)))
print(mu0)
print(mu1)
print(list(idx0))
print(list(idx1))
print("idx0", idx0)
print("idx1", idx1)
print("kidx0", U1[kidx0])
print("kidx1", U1[kidx1])

print(U0)
print(U1)
