import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt
import pickle
import scipy.optimize
"""
This script transforms the projected boundary using the basis defined in the
plane. The projected boundary is then rotated in the plane to minimize its
bounding box.

"""

inputfile = sys.argv[1]
outputfile = sys.argv[2]

if len(sys.argv) < 4:
    figfile = None
else:
    figfile = sys.argv[3]
if len(sys.argv) < 5:
    showplot = 0
else:
    showplot = int(sys.argv[4])

def objective_function(xy, mu, theta):
    try:
        theta = theta[0]
    except:
        pass
    rxy = sf.fitting.rotate2(xy, mu, theta)
    bbox = sf.fitting.bbox2(rxy)
    return sf.fitting.bbox2_vol(bbox)

data = pickle.load(open(inputfile, 'rb'))

# Rotate data into new coordinate system
T = data.proj_basis
bnd_xy = T.T.dot(data.bnd_xyz.T).T



center = sf.fitting.mean(bnd_xy) 
center = np.tile(center, (bnd_xy.shape[0],1)) 

obj = lambda theta : objective_function(bnd_xy, center, theta)

var = scipy.optimize.minimize(obj, (0.0,), method='Nelder-Mead')['x']
data.theta = var[0]
data.center = center
rxy = sf.fitting.rotate2(bnd_xy, center, data.theta)
data.bnd_rxy = rxy
data.proj_xy = bnd_xy
data.bnd_rz = data.bnd_proj_xyz[:,2]

plt.plot(bnd_xy[:,0], bnd_xy[:,1], 'bo')
plt.plot(rxy[:,0], rxy[:,1], 'ro')
bbox = sf.fitting.bbox2(bnd_xy)
helper.plot_points2(helper.close_boundary(bbox),'b')
bbox = sf.fitting.bbox2(rxy)
helper.plot_points2(helper.close_boundary(bbox),'r')
plt.legend(['Before rotation', 'After rotation'])
plt.savefig(figfile)
helper.show(showplot)
pickle.dump(data, open('.'.join(outputfile.split('.')[:-1]) + '.p', 'wb'))
