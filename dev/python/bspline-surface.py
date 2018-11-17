import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
outputfile = sys.argv[2]
invert_mapping = int(sys.argv[3])

if len(sys.argv) < 4:
    vtkfile = None
else:
    vtkfile = sys.argv[4]

if len(sys.argv) < 5:
    jsonfile = None
else:
    jsonfile = sys.argv[5]

if len(sys.argv) < 6:
    uvfig = None
else:
    uvfig = sys.argv[6]

if len(sys.argv) < 7:
    showplot = 0
else:
    showplot = int(sys.argv[7])


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

def mapping(S, bnd, data, x, y, invert=0, st=10):
    print("Mapping (x, y) coordinates to (u, v) coordinates")
    u = sf.bspline.xmap(x)
    v = sf.bspline.xmap(y)
    if invert:
        l, r, b, t = bnd
        for i in range(len(x)):
            u[i], v[i] = sf.bspline.uvinv(x[i], y[i], u[i], v[i], l, r, b, t)
            if i % st == 0:
                print("%d out of = %d points completed" % (i, len(x)))

    def force_boundaries():
        """
        Force boundary points to map the sides of the UV grid.
        """
        for idx in data.top_ids:
            u[idx] = 1
        for idx in data.bottom_ids:
            u[idx] = 0
        for idx in data.left_ids:
            v[idx] = 0
        for idx in data.right_ids:
            v[idx] = 1

        
    if uvfig:
        plt.plot(u, v, 'bo')
        plt.xlabel('u')
        plt.ylabel('v')
        plt.savefig(uvfig)
        print("Wrote", uvfig)
    return u, v


def fit_surface(S, bnd, data, x, y, z):
    u, v = mapping(S, bnd, data, x, y, invert_mapping)
    S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv, data.corner_ids)
    return S

def plot_transfinite(S, bnds):
    left, right, bottom, top = bnds
    ax = helper.plot_grid(S.X, S.Y, 0*S.Z)
    ax = helper.plot_grid(S.Px, S.Py, 0*S.Pz, ax, color='C0')
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


int_knot_v = left.int_knot
int_knot_u = bottom.int_knot

pu = bottom.p
pv = left.p

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
S = fit_surface(S, bnds, data, x, y, z)
S.eval(20,20)

u = np.linspace(0, 1.0, 40)
v = np.linspace(0, 1.0, 40)
r = 0
px =  Px[0]
py =  Py[0]

plot_transfinite(S, bnds)
helper.show(showplot)

sf.vtk.write_surface(vtkfile, S.X, S.Y, S.Z)

S.rwPx, S.rwPy, S.rwPz, data.coords = restore(data, S.Px, S.Py, S.Pz, data.pcl_xyz)
X, Y, Z, data.coords = restore(data, S.X, S.Y, S.Z, data.pcl_xyz)
ax = helper.plot_grid(S.X, S.Y, S.Z)
helper.plot_points(data.coords, ax=ax, style='ro')
sf.vtk.write_surface(vtkfile, X, Y, Z)
print("Wrote:", vtkfile)
S.json(jsonfile)
print("Wrote:", jsonfile)

pickle.dump((S, data), open(outputfile, 'wb'))
