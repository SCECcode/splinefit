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

cfg = pickle.load(open(inputfile, 'rb'))
bnds = cfg[0:4]
data = cfg[4]

def rotate(data):
    # Rotate point cloud
    T = data.proj_basis
    xy = T.T.dot(data.pcl_xyz.T).T
    center = sf.fitting.mean(xy) 
    center = np.tile(center, (xy.shape[0],1)) 
    rxy = sf.fitting.rotate2(xy, center, data.theta)
    return rxy

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


unpack = lambda bnd_xy : [a for subbnd in bnd_xy for a in subbnd]

nu = 50
nv = 25

num_uv = [nv, nv, nu, nu]

bnd_xy = []
for bi, ni in zip(bnds, num_uv):
    bnd_xy.append(helper.evalcurve(bi, ni))

#for bi in bnds:
#    bnd_xy.append([bi.px, bi.py])

bnd = unpack(bnd_xy)

xl, yl, xr, yr, xb, yb, xt, yt = bnd
X, Y = build_grid(bnd, nu, nv)
helper.plot_grid(X, Y)
plt.show()
