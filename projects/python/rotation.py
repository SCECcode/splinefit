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
xy = T.T.dot(data.bnd_xyz.T).T


center = sf.fitting.mean(xy) 
center = np.tile(center, (xy.shape[0],1)) 

obj = lambda theta : objective_function(xy, center, theta)

var = scipy.optimize.minimize(obj, (0.0,), method='Nelder-Mead')['x']
data.theta = var[0]
data.center = center
rxy = sf.fitting.rotate2(xy, center, data.theta)
data.rxy = rxy
data.proj_xy = xy

plt.plot(xy[:,0], xy[:,1], 'bo')
plt.plot(rxy[:,0], rxy[:,1], 'ro')
bbox = sf.fitting.bbox2(xy)
helper.plot_points2(helper.close_boundary(bbox),'b')
bbox = sf.fitting.bbox2(rxy)
helper.plot_points2(helper.close_boundary(bbox),'r')
plt.legend(['Before rotation', 'After rotation'])
plt.savefig(figfile)
pickle.dump(data, open('.'.join(outputfile.split('.')[:-1]) + '.p', 'wb'))
