import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt


num_knots=48
npts=100
npts2=300
a=0
b=1
degree=3
px = np.linspace(a, b, npts)
py = np.sin(65*px)*np.exp(-px**2) + 0.00 * np.random.randn(npts)
px = np.cos(4.2*px)

curve1 = helper.Struct()
curve1.x = px
curve1.y = py
curve1.p = degree
curve1.U = sf.bspline.uniformknots(num_knots, degree, a=a, b=b)
curve1.s = np.linspace(a,b, npts)#sf.bspline.chords(px, py, a=a, b=b)
curve1.U = sf.bspline.kmeansknots(curve1.s, num_knots, degree, a=a, b=b)
curve1.Px, curve1.Py, res = sf.bspline.lsq2(curve1.s, curve1.x, curve1.y, curve1.U, curve1.p)

curve2 = helper.Struct()
curve2.x = px
curve2.y = py
curve2.p = degree
curve2.U = sf.bspline.uniformknots(num_knots, degree, a=a, b=b)
curve2.s = sf.bspline.chords(px, py, a=a, b=b)
curve2.Px = curve1.Px*0
curve2.Py = curve1.Py*0

curve2.Px[0] = px[0]
curve2.Py[0] = py[0]
curve2.Px[-1] = px[-1]
curve2.Py[-1] = py[-1]


curve2.Px[1] = px[10]
curve2.Py[1] = py[10]
curve2.Px[2] = px[20]
curve2.Py[2] = py[20]
curve2.Px[3] = px[30]
curve2.Py[3] = py[30]

def nearest():
    points = np.stack((px, py))
    i = sf.fitting.argnearest(points, p)


print(curve1.U)
print(curve1.Px)
print(curve2.U.shape)
print(curve2.Px.shape)
print(curve2.Py.shape)

cx, cy = helper.evalcurve(curve1, npts2)
cx2, cy2 = helper.evalcurve(curve2, npts2)
plt.plot(px,py,'k') 
plt.plot(cx,cy,'g') 
plt.plot(curve1.Px, curve1.Py, 'go-', alpha=0.3)
#plt.plot(cx2,cy2,'b') 
#plt.plot(curve2.Px, curve2.Py, 'bo-', alpha=0.3)
plt.show()
                   


