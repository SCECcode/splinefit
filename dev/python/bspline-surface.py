import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import mesh
import matplotlib.pyplot as plt
import bspline_surface_functions as bsf

inputfile = sys.argv[1]
outputfile = sys.argv[2]
invert_mapping = int(sys.argv[3])
surf_smooth = float(sys.argv[4])
regularization = float(sys.argv[5])
use_mapping = float(sys.argv[6])
padding = 0.5

if len(sys.argv) < 7:
    vtkfile = None
else:
    vtkfile = sys.argv[7]

if len(sys.argv) < 8:
    jsonfile = None
else:
    jsonfile = sys.argv[8]

if len(sys.argv) < 9:
    uvfig = None
else:
    uvfig = sys.argv[9]

if len(sys.argv) < 10:
    showplot = 0
else:
    showplot = int(sys.argv[10])

cfg = pickle.load(open(inputfile, 'rb'))
bnds = cfg[0:4]
data = cfg[4]
bnd = bsf.build_cv_boundary(bnds)
xl, yl, xr, yr, xb, yb, xt, yt = bnd
pcl = bsf.rotate(data)

if use_mapping:
    nu = len(xb)
    nv = len(xl)
    left, right, bottom, top = bnds
    pu = bottom.p
    pv = left.p
else:
    nu = len(xb)
    nv = len(xl)
    nu = 60
    nv = 60
    pu = 3
    pv = 3

if use_mapping:
    X, Y, bnd = bsf.build_grid(bnd, nu, nv)
    #U = bottom.U
    #V = left.U
    int_knot_u = sf.bspline.numknots(nu, pu, interior=1)
    int_knot_v = sf.bspline.numknots(nv, pv, interior=1)
    U = sf.bspline.uniformknots(int_knot_u, pu, a=0, b=1)
    V = sf.bspline.uniformknots(int_knot_v, pv, a=0, b=1)
    print("m, p, n", len(U), pu, X.shape[0])
else:
    u, v, X, Y = bsf.background_grid(data, nu, nv, pu, pv, padding=padding)
    int_knot_u = sf.bspline.numknots(nu, pu, interior=1)
    int_knot_v = sf.bspline.numknots(nv, pv, interior=1)
    U = sf.bspline.uniformknots(int_knot_u, pu, a=-padding, b=1 + padding)
    V = sf.bspline.uniformknots(int_knot_v, pv, a=-padding, b=1 + padding)

S = sf.bspline.Surface(U, V, pu, pv, X, Y, 0*X, label='grid')
print(S)

# Project points onto triangulation
qpoints = np.vstack((X.flatten(), Y.flatten(), 0*S.Pz.flatten())).T
tri, qpoints = mesh.project(pcl, qpoints)
S.Pz = np.reshape(qpoints[:,2], (X.shape[0], Y.shape[1]))

if use_mapping:
    S, data.u, data.v = bsf.fit_surface(S, bnds, data, qpoints, surf_smooth, 
                                        invert_mapping, uvfig, fit=False)
else:
    pass
    # Least squares fitting
    #S = bsf.fit_background_surface(S, data, qpoints, surf_smooth=surf_smooth,
    #        uvfig=uvfig)


S.rwPx, S.rwPy, S.rwPz = sf.fitting.restore(S.Px, S.Py, S.Pz,
        data.basis, data.mu, data.std, data.center, data.theta)
#S.X, S.Y, S.Z = sf.fitting.restore(S.X, S.Y, S.Z, data.basis, data.mu,
#                                   data.std, data.center, data.theta)
rwx, rwy, rwz = sf.fitting.restore(pcl[:,0], pcl[:,1], pcl[:,2], data.basis,
                                   data.mu, data.std, data.center, data.theta)
coords = np.vstack((rwx, rwy, rwz)).T


# Smooth out surface
#for i in range(1):
#    S.Pz = bsf.diffuse(S.Pz, 0.1, nsteps=10)

S.eval(20, 20, rw=0)
ax = bsf.plot_grid(S, pcl)
plt.triplot(pcl[:,0], pcl[:,1], tri, 'k-', alpha=0.6)
helper.plot_points(qpoints, ax, 'r*')
helper.show(showplot)

left, right, bottom, top = bnds
labels = ['left', 'right', 'bottom', 'top']
curves = {}
for i, bnd in enumerate(bnds):
    c = sf.bspline.Curve(bnd.U, bnd.p, bnd.Px, bnd.Py, bnd.Pz,
                                         label=labels[i])
    c.rwPx, c.rwPy, c.rwPz = sf.fitting.restore(bnd.Px, bnd.Py, bnd.Pz,
                data.basis, data.mu, data.std, data.center, data.theta)
    x, y, z = c.eval(npts=40, rw=0)
    curves[labels[i]] = c
    helper.plot_curve(x, y, z, ax, 'C%d-' % i)
    filename = '.'.join(jsonfile.split('.')[:-1]) + '_%s.json' % c.label
    c.json(filename)
    print("Wrote:", filename)


helper.show(showplot)

S.eval(80, 80, rw=1)
sf.vtk.write_surface(vtkfile, S.X, S.Y, S.Z)
print("Wrote:", vtkfile)
S.json(jsonfile)
print("Wrote:", jsonfile)

data.boundary_curves = curves
pickle.dump((S, data), open(outputfile, 'wb'))
