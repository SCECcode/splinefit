import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt
import pickle
import scipy.optimize
"""
This script fits the minimum area quadrilateral to the boundary points. All
boundary points must be contained inside it.

"""

inputfile = sys.argv[1]
outputfile = sys.argv[2]

if len(sys.argv) < 4:
    figfile = None
else:
    figfile = sys.argv[3]

def objective_fun(p):
    """
    Minimize the area of the quad defined by the four vertices p0, p1, p2, p3.

    """
    p0, p1, p2, p3 = vars_to_coords(p)
    Q1 = sf.fitting.quad_vol(p0, p1, p2, p3)
    T1 = max(sf.fitting.triangle_vol(p0, p2, p3),
             sf.fitting.triangle_vol(p0, p2, p3))
    area =  max(Q1, T1) 
    return area

def vars_to_coords(p):
    p0 = np.array([p[0], p[1]])
    p1 = np.array([p[2], p[3]])
    p2 = np.array([p[4], p[5]])
    p3 = np.array([p[6], p[7]])
    return p0, p1, p2, p3

def coords_to_vars(p0, p1, p2, p3):
    return [p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]

def constraint(xy, p):
    """
    Require that all points lie inside the quad. All points lie inside the quad
    if the signed distance function, for each point, is non-positive. 
    """
    p0, p1, p2, p3 = vars_to_coords(p)

    d = lambda x, y : sf.fitting.sdquad2(np.array((x, y)), p0, p1, p2, p3) 
    dout = -1e3
    for i in range(xy.shape[0]):
        dtemp = d(xy[i,0], xy[i,1])
        if dtemp > dout:
            dout = dtemp
    return -dout

data = pickle.load(open(inputfile, 'rb'))
bbox = sf.fitting.bbox2(data.rxy)

p0 = bbox[0,:] 
p1 = bbox[1,:]
p2 = bbox[2,:]
p3 = bbox[3,:]

cons = ({'type' : 'ineq', 'fun' : lambda x : constraint(data.rxy, x)},)
p = coords_to_vars(p0, p1, p2, p3)
A0 = objective_fun(p)
print("Initial area of bounding box:", A0)
ans = scipy.optimize.minimize(objective_fun, p, method='SLSQP', 
        constraints=cons, options={'maxiter' : 1e3})
Amin = ans.fun
print("Area after optimization:", Amin)
if Amin > A0:
    print("Optimization failed!")

p0, p1, p2, p3 = vars_to_coords(ans.x)
quad = np.reshape(ans.x, (4, 2))

data.quad = quad
bbox = sf.fitting.bbox2(data.rxy)
helper.plot_points2(helper.close_boundary(bbox),'b')
helper.plot_points2(helper.close_boundary(quad),'r')
plt.plot(data.rxy[:,0], data.rxy[:,1], 'ko')
plt.legend(['Before optimization', 'After optimization'])
plt.savefig(figfile)
pickle.dump(data, open('.'.join(outputfile.split('.')[:-1]) + '.p', 'wb'))
