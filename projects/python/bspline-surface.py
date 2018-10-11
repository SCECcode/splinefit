import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
outputfile = sys.argv[2]
sm = float(sys.argv[3])

if len(sys.argv) < 4:
    vtkfile = None
else:
    vtkfile = sys.argv[4]


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
    center = np.tile(data.center[0,:], (xy.shape[0],1)) 
    rxy = sf.fitting.rotate2(xy, center, data.theta)
    xyz[:,0:2] = rxy
    return xyz

def restore(data, X, Y, Z, pcl_xyz):
    """
    Rotate surface back to the original coordinate system.
    Also, remove normalization

    """
    T = data.proj_basis

    nx = X.shape[0]
    ny = X.shape[1]
    xy = np.vstack((X.flatten(), Y.flatten())).T
    center = np.tile(data.center[0,:], (xy.shape[0],1)) 
    rxy = sf.fitting.rotate2(xy, center, -data.theta)
    xyz = np.vstack((rxy.T, Z.flatten())).T
    xyz = data.basis.dot(xyz.T).T
    xyz = sf.fitting.renormalize(xyz, data.mu, data.std)
    X = np.reshape(xyz[:,0], (nx, ny))
    Y = np.reshape(xyz[:,1], (nx, ny))
    Z = np.reshape(xyz[:,2], (nx, ny))
    coords = sf.fitting.renormalize(pcl_xyz, data.mu, data.std)
    
    #xyz[:,0:2] = rxy
    return X, Y, Z, coords


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
        bnd_xy.append([bi.Px, bi.Py])
    bnd = unpack(bnd_xy)
    return bnd

def build_grid(bnd, nu, nv):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)
    status, bnd = sf.transfinite.fixboundaries(*bnd)
    bnd = sf.transfinite.fixcorners(*bnd)
    assert sf.transfinite.checkcorners(*bnd)
    #assert sf.transfinite.checkboundaries(*bnd)
    X, Y = sf.transfinite.bilinearinterp(*bnd, U, V)
    return X, Y

def fit_surface(S, bnd, x, y, z):
    u = sf.bspline.xmap(x)
    v = sf.bspline.xmap(y)

    l, r, b, t = bnd

    #u = np.linspace(0, 1, 20)
    #v = np.linspace(0, 1, 20)
    #u, v = np.meshgrid(u, v)
    #u = u.flatten()
    #v = v.flatten()

    print("Mapping (x, y) coordinates to (u, v) coordinates")
    for i in range(len(u)):
        u[i], v[i] = sf.bspline.uvinv(x[i], y[i], u[i], v[i], l, r, b, t)
        print("%d out of = %d points completed" % (i, len(u)))
    plt.plot(u, v, 'bo')
    plt.show()
    S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv)
    return S

def plot_transfinite(S, bnds):
    left, right, bottom, top = bnds
    ax = helper.plot_grid(S.X, S.Y, S.Z)
    ax = helper.plot_grid(S.Px, S.Py, 0*S.Pz, ax, color='red')
    ax.plot(pcl[:,0], pcl[:,1],'b*')
    ax.plot(xl, yl,'k-')
    ax.plot(xr, yr,'k-')
    ax.plot(xt, yt,'k-')
    ax.plot(xb, yb,'k-')
    cx, cy = helper.evalcurve(top, 100)
    ax.plot(cx,cy,'b-')
    cx, cy = helper.evalcurve(bottom, 100)
    ax.plot(cx,cy,'b-')
    cx, cy = helper.evalcurve(left, 100)
    ax.plot(cx,cy,'b-')
    cx, cy = helper.evalcurve(right, 100)
    ax.plot(cx,cy,'b-')
    plt.show()


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
xl, yl, xr, yr, xb, yb, xt, yt = bnd
left, right, bottom, top = bnds
nu = len(xb)
nv = len(xl)
pcl = rotate(data)
X, Y = build_grid(bnd, nu, nv)


plt.show()
int_knot_v = left.int_knot
int_knot_u = bottom.int_knot

pu = bottom.p
pv = left.p
print(pu, pv)

U = sf.bspline.uniformknots(int_knot_u, bottom.p)
V = sf.bspline.uniformknots(int_knot_v, left.p)
U = bottom.U
V = left.U

Px = X
Py = Y

x = pcl[:,0]
y = pcl[:,1]
z = pcl[:,2]
S = sf.bspline.Surface(U, V, pu, pv, Px, Py, 0*Px)
S = fit_surface(S, bnds, x, y, z)
S.eval(20,20)

u = np.linspace(0, 1.0, 40)
v = np.linspace(0, 1.0, 40)
r = 0
px =  Px[0]
py =  Py[0]

print(U, V)
plot_transfinite(S, bnds)

#helper.plot_points(pcl, ax=ax, style='ro')
sf.vtk.write_surface(vtkfile, S.X, S.Y, S.Z)

X, Y, Z, data.coords = restore(data, S.X, S.Y, S.Z, data.pcl_xyz)
ax = helper.plot_grid(S.X, S.Y, S.Z)
helper.plot_points(data.coords, ax=ax, style='ro')
sf.vtk.write_surface(vtkfile, X, Y, Z)

#ax = helper.plot_grid(X, Y, 0*X)
# TODO: Build surface struct
#pickle.dump((X, Y, Z, data), open(outputfile, 'wb'))
print("Wrote:", vtkfile)
