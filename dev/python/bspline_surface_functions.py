import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt

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

def unpack(bnd_xy):
    return [a for subbnd in bnd_xy for a in subbnd]

def build_cv_boundary(bnds):
    bnd_xy = []
    for bi in bnds:
        bnd_xy.append([bi.Px, bi.Py])
    bnd = unpack(bnd_xy)
    return bnd

def mapping(S, bnd, data, x, y, invert=0, st=10, uvfig=None):
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

def fit_surface(S, bnds, data, pcl, surf_smooth=False, invert_mapping=False,
                uvfig=None, fit=True):
    x = pcl[:,0]
    y = pcl[:,1]
    z = pcl[:,2]
    u, v = mapping(S, bnds, data, x, y, invert_mapping, uvfig)
    if fit:
        S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv,
                data.corner_ids, s=surf_smooth)
    clamp_boundaries(data, bnds, u, v, x, y, z, S, surf_smooth)
    return S, u, v

def fit_background_surface(S, data, pcl, surf_smooth=0, uvfig=None,
                           padding=0.0):
    x = pcl[:,0]
    y = pcl[:,1]
    z = pcl[:,2]
    u = sf.bspline.xmap(x)
    v = sf.bspline.xmap(y)

    if uvfig:
        plt.plot(u, v, 'bo')
        plt.xlabel('u')
        plt.ylabel('v')
        plt.savefig(uvfig)
        print("Wrote", uvfig)
    S.Pz, res = sf.bspline.lsq2surf(u, v, z, S.U, S.V, S.pu, S.pv,
                                    s=surf_smooth)
    return S

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

def background_grid(data, nu, nv, pu, pv, padding=0):
    """
        Construct background grid by using bounding box and padding the bounding
        box so that no boundary basis functions are needed.
        Bounding box coordinates are: bottom left,
        then bottom right, top right, and top left coordinate.
    """

    # UV-grid
    data.u = np.linspace(0, 1, nu)
    data.v = np.linspace(0, 1, nv)
    data.hu = data.u[1] - data.u[0]
    data.hv = data.v[1] - data.v[0]

    # Grid
    b = data.bbox
    Lx = b[1][0] - b[0][0]
    Ly = b[2][1] - b[0][1]
    data.Gx = np.linspace(b[0][0] - padding*Lx, b[1][0] + padding*Lx, nu)
    data.Gy = np.linspace(b[0][1] - padding*Ly, b[2][1] + padding*Ly, nv)

    U, V = np.meshgrid(data.u, data.v)
    X, Y = np.meshgrid(data.Gx, data.Gy)
    return U, V, X, Y

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

def clamp_boundaries(data, bnds, u, v, x, y, z, S, regularization):
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

def misfit():
#FIXME: Implement misfit check
#x, y, z = S.surfacepoints(data.u, data.v, data.coords)
#rx = x-data.coords[:, 0]
#ry = y-data.coords[:, 1]
#rz = z-data.coords[:, 2]
#
#dist = np.sqrt(rx**2 + ry**2 + rz**2)
#helper.plot_points(coords, ax=ax, style='bo')
#mshfile = '.'.join(vtkfile.split('.')[0:-1]) + '_error.vtk'
#sf.vtk.write_triangular_mesh(mshfile, data.coords, data.tris)
#print("Computing misfit")
#err = S.compute_misfit(data.u, data.v, data.coords)
#sf.vtk.append_scalar(mshfile, err, label='error')
    pass

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

def plot_grid(S, pcl, ax=None):
    x = pcl[:,0]
    y = pcl[:,1]
    z = pcl[:,2]
    if not ax: 
        ax = helper.plot_grid(S.X, S.Y, S.Z)
    else:
        helper.plot_grid(S.X, S.Y, S.z, ax=ax)
    coords = np.vstack((x, y, z)).T
    helper.plot_points(coords, style='go', ax=ax)
    return ax

def diffuse(u, cfl, nsteps=100):
    v = np.ma.masked_equal(u, 0.0)
    v.mask = ~v.mask
    v.mask = 0*u + 1
    v.mask[:,0] = 0
    v.mask[:,-1] = 0
    v.mask[-1,:] = 0
    v.mask[0,:] = 0
    v = u

    for i in range(nsteps):
        imask = v
        #imask = v.copy()
        #imask.mask = imask.mask
        v = v - cfl*                                                           \
                (
                - np.roll(imask, 1, 0) 
                - np.roll(imask, -1, 0) 
                + 4*v 
                - np.roll(imask, 1, 1) 
                - np.roll(imask, -1, 1)
                )
    return np.array(v)
