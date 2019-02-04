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
surf_smooth = float(sys.argv[4])
regularization = float(sys.argv[5])

if len(sys.argv) < 6:
    vtkfile = None
else:
    vtkfile = sys.argv[6]

if len(sys.argv) < 7:
    jsonfile = None
else:
    jsonfile = sys.argv[7]

if len(sys.argv) < 8:
    uvfig = None
else:
    uvfig = sys.argv[8]

if len(sys.argv) < 9:
    showplot = 0
else:
    showplot = int(sys.argv[9])


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
    return X, Y, bnd

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


def fit_surface(S, bnds, data, x, y, z, pcl, surf_smooth):
    u, v = mapping(S, bnds, data, x, y, invert_mapping)
    S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv,
            data.corner_ids, s=surf_smooth)
    clamp_boundaries(bnds, u, v, x, y, z, S)
    return S, u, v

def boundary_points(bnd, x, y, z, u, v):
    pcl = np.vstack((x,y,0*z)).T
    xout = []
    yout = []
    zout = []
    uout = []
    vout = []
    for i in range(bnd.shape[0]):
        b = [bnd[i,0], bnd[i,1], 0]
        idx = sf.fitting.argnearest(pcl,b)
        xout.append(x[idx])
        yout.append(y[idx])
        zout.append(z[idx])
        uout.append(u[idx])
        vout.append(v[idx])

    return np.array(xout), np.array(yout), np.array(zout), np.array(uout), \
           np.array(vout)


def flip(x1, x2, x3, x4, x5):
    if x1[0] > x1[-1]:
        return x1[::-1], x2[::-1], x3[::-1], x4[::-1], x5[::-1]
    else:
        return x1, x2, x3, x4, x5

def flip_p(bnd):
    bnd.Px = bnd.Px[::-1]
    bnd.Py = bnd.Py[::-1]
    bnd.Pz = bnd.Pz[::-1]
    return bnd


def clamp_boundaries(bnds, u, v, x, y, z, S):
    xb, yb, zb, ub, vb = boundary_points(data.bottom, x, y, z, u, v)
    xt, yt, zt, ut, vt = boundary_points(data.top, x, y, z, u, v)
    xl, yl, zl, ul, vl = boundary_points(data.left, x, y, z, u, v)
    xr, yr, zr, ur, vr = boundary_points(data.right, x, y, z, u, v)

    xb, yb, zb, ub, vb = flip(xb, yb, zb, ub, vb)
    xt, yt, zt, ut, vt = flip(xt, yt, zt, ut, vt)
    yl, xl, zl, ul, vl = flip(yl, xl, zl, ul, vl)
    yr, xr, zr, ur, vr = flip(yr, xr, zr, ur, vr)

    print("Clamping boundaries")
    print("Number of points Bottom:", xb.shape[0])
    print("Number of points Top:", xt.shape[0])
    print("Number of points Left:", xl.shape[0])
    print("Number of points Right:", xr.shape[0])
    print("Regularization:", regularization)
    print("U,V", S.U.shape, S.V.shape)

    left, right, bottom, top = bnds
    if top.Px[0] > top.Px[-1]:
        top = flip_p(top)
    S.Px[-1,:] = top.Px
    S.Py[-1,:] = top.Py
    S.Pz[-1,:] = top.Pz

    if bottom.Px[0] > bottom.Px[-1]:
        bottom = flip_p(bottom)
    S.Px[0,:] = bottom.Px
    S.Py[0,:] = bottom.Py
    S.Pz[0,:] = bottom.Pz

    if left.Py[0] > left.Py[-1]:
        left = flip_p(left)
    S.Px[:,0] = left.Px
    S.Py[:,0] = left.Py
    S.Pz[:,0] = left.Pz

    if right.Py[0] > right.Py[-1]:
        right = flip_p(right)
    S.Px[:,-1] = right.Px
    S.Py[:,-1] = right.Py
    S.Pz[:,-1] = right.Pz


def plot_transfinite(S, bnds):
    left, right, bottom, top = bnds
    ax = helper.plot_grid(S.X, S.Y, S.Z)
    ax = helper.plot_grid(S.Px, S.Py, S.Pz, ax, color='C0')
    ax.plot(pcl[:,0], pcl[:,1], pcl[:,2],'b*')
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
X, Y, bnd = build_grid(bnd, nu, nv)


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
S, data.u, data.v = fit_surface(S, bnds, data, x, y, z, pcl, surf_smooth)
data.z = z
S.eval(80,80)

print("U degree:", S.pu)
print("V degree:", S.pv)
print("num Px", Px.shape)
print("num U", U.shape)


u = np.linspace(0, 1.0, 40)
v = np.linspace(0, 1.0, 40)
r = 0
px = Px[0]
py = Py[0]

#plot_transfinite(S, bnds)
#helper.show(showplot)

#plt.clf()
#ax = helper.plot_grid(S.X, S.Y, S.Z)
#helper.plot_points(data.coords, ax=ax, style='ro')
#
coords = np.vstack((x, y, z)).T
ax = helper.plot_points(coords, style='bo')

pts = np.vstack((left.x, left.y, left.z)).T
helper.plot_points(pts, ax=ax, style='C1o-')
pts = np.vstack((right.x, right.y, right.z)).T
helper.plot_points(pts, ax=ax, style='C2o-')
pts = np.vstack((top.x, top.y, top.z)).T
helper.plot_points(pts, ax=ax, style='C3o-')
pts = np.vstack((bottom.x, bottom.y, bottom.z)).T
helper.plot_points(pts, ax=ax, style='C4o-')
coords = np.vstack((data.bnd_rxy[:,0], data.bnd_rxy[:,1], data.bnd_rz)).T
helper.plot_points(coords, style='ko', ax=ax)
helper.show(showplot)


S.rwPx, S.rwPy, S.rwPz, data.coords = restore(data, S.Px, S.Py, S.Pz, data.pcl_xyz)
S.X, S.Y, S.Z, data.coords = restore(data, S.X, S.Y, S.Z, data.pcl_xyz)
x, y, z = S.surfacepoints(data.u, data.v, data.coords)


rx = x-data.coords[:, 0]
ry = y-data.coords[:, 1]
rz = z-data.coords[:, 2]

dist = np.sqrt(rx**2 + ry**2 + rz**2)
helper.plot_points(coords, ax=ax, style='bo')

sf.vtk.write_surface(vtkfile, S.X, S.Y, S.Z)
mshfile = '.'.join(vtkfile.split('.')[0:-1]) + '_error.vtk'
sf.vtk.write_triangular_mesh(mshfile, data.coords, data.tris)
print("Computing misfit")
err = S.compute_misfit(data.u, data.v, data.coords)
sf.vtk.append_scalar(mshfile, err, label='error')
print("Wrote:", mshfile)
print("Wrote:", vtkfile)
S.json(jsonfile)
print("Wrote:", jsonfile)

pickle.dump((S, data), open(outputfile, 'wb'))
