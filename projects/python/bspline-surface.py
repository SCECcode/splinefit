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

if len(sys.argv) < 5:
    vtkfile = None
else:
    vtkfile = sys.argv[5]


def rotate(data):
    """
    Rotate point cloud using the basis vectors of the best fitting plane, and
    bounding box.
    """
    # Rotate point cloud
    T = data.proj_basis
    xy = T.T.dot(data.pcl_xyz.T).T
    xyz = data.basis.T.dot(data.pcl_xyz.T).T
    center = sf.fitting.mean(xy) 
    center = np.tile(center, (xy.shape[0],1)) 
    rxy = sf.fitting.rotate2(xy, center, data.theta)
    xyz[:,0:2] = rxy
    return xyz

def unpack(bnd_xy):
    return [a for subbnd in bnd_xy for a in subbnd]

def init_interp(bnds):
    """
    Prepare for transfinite interpolation by evaluating the boundary curves
    using freely chosen number of u, v points.

    """
    nu = 50
    nv = 25
    
    num_uv = [nv, nv, nu, nu]
    
    bnd_xy = []
    for bi, ni in zip(bnds, num_uv):
        bnd_xy.append(helper.evalcurve(bi, ni))
    bnd = unpack(bnd_xy)
    return bnd

def build_cv_boundary(bnds):
    bnd_xy = []
    for bi in bnds:
        bnd_xy.append([bi.px, bi.py])
    bnd = unpack(bnd_xy)
    return bnd

def build_grid(bnd, nu, nv):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)
    status, bnd = sf.transfinite.fixboundaries(*bnd)
    bnd = sf.transfinite.fixcorners(*bnd)
    assert sf.transfinite.checkcorners(*bnd)
    assert sf.transfinite.checkboundaries(*bnd)
    X, Y = sf.transfinite.bilinearinterp(*bnd, U, V)
    return X, Y

def fit_surface(p, x, y, z, U, V):
    u = sf.bspline.xmap(x)
    v = sf.bspline.xmap(y)
    Pz, res = sf.bspline.lsq2surf(u, v, z, U, V, p)
    return Pz

def add_zeros2(Px, p):
    """
    Control points should have zeros at the end of the interval
    """
    px = np.zeros((Px.shape[0] + p, Px.shape[1] + p))
    px[:Px.shape[0], :Px.shape[1]] = Px
    return px




cfg = pickle.load(open(inputfile, 'rb'))
bnds = cfg[0:4]
data = cfg[4]
bnd = build_cv_boundary(bnds)
#bnd = init_interp(bnds)
xl, yl, xr, yr, xb, yb, xt, yt = bnd
nu = len(xb)
nv = len(xl)
pcl = rotate(data)
X, Y = build_grid(bnd, nu, nv)
p = 3
nu = X.shape[0] -2*p-1 
nv = X.shape[1] -2*p-1
int_knot_v = bnds[0].int_knot # left
int_knot_u = bnds[2].int_knot # bottom

U = sf.bspline.uniformknots(int_knot_u, p)
V = sf.bspline.uniformknots(int_knot_v, p)

Px = X
Py = Y
Px = add_zeros2(Px, p)
Py = add_zeros2(Py, p)

x = pcl[:,0]
y = pcl[:,1]
z = pcl[:,2]
Pz = fit_surface(p, x, y, z, U, V)

nu = 9
nv = 9

u = np.linspace(0, 1.0, 37)
v = np.linspace(0, 1.0, 37)
r = 0
px =  Px[0]
py =  Py[0]
X = sf.bspline.evalsurface(p, U, V, Px, u, v)
Y = sf.bspline.evalsurface(p, U, V, Py, u, v)
Z = sf.bspline.evalsurface(p, U, V, Pz, u, v)
ax = helper.plot_grid(X, Y, Z)
helper.plot_points(pcl, ax=ax, style='ro')
sf.vtk.write_surface(vtkfile, X, Y, Z)
plt.show()
print("Wrote:", vtkfile)
