"""
Compute the distance from the original set of coordinates to the fitted Bspline
surface

"""

import warnings
import sys
import pickle
import splinefit as sf
from scipy.optimize import minimize

inputfile = sys.argv[1]
outputfile = sys.argv[2]
vtkfile = sys.argv[3]

S, data = pickle.load(open(inputfile, 'rb'))

def run(verbose=0, ord=2, debug=0):
    print("Finding minimum distance from point cloud to fitted surface.") 
    i = 0
    numpts = len(data.u)
    dists = data.u*0
    u = data.u*0
    v = data.v*0
    points = 0*data.pcl_xyz
    for ui, vi, pi in zip(data.u, data.v, data.pcl_xyz):
        if verbose:
            print("Point: %d out of %d" % (i, numpts))
        if debug:
            print("Minimizing for point %d: (%g %g %g) -> S(%g, %g)" % 
                  (i, pi[0], pi[1], pi[2], ui, vi))
        obj = lambda x : S.distance(x[0], x[1], pi, ord=ord, coordinates="local")
        #res = minimize(obj, [ui, vi], method='SLSQP', tol=1e-12)
        res = minimize(obj, [0, 0], method='SLSQP', tol=1e-12)
        local_dist = res['fun']
        u[i] = res['x'][0]
        v[i] = res['x'][1]
        print(ui, vi, u[i], v[i], local_dist)
        #print(u[i]-ui, v[i]-vi)
        dists[i] = S.distance(u[i], v[i], data.coords[i,:], ord=ord,
                                  coordinates="global")
        points[i,:] = data.coords[i,:]#S.distance(u[i], v[i], data.coords[i,:], ord=ord,
                                 #surf_point=1, coordinates="global")
        if not res['success']:
            print("Failed!")
            warnings.warn('Minimization failed')
        else:
            points[i,:] = S.distance(u[i], v[i], data.coords[i,:], ord=ord,
                                     surf_point=1, coordinates="global")
        # if debug:
        #     print("Distance:", dists[i], "Success:", res['success'])
        i += 1
    return dists, points


dists, points = run(verbose=1)

sf.vtk.write_triangular_mesh(vtkfile, data.coords, data.tris)
sf.vtk.append_scalar(vtkfile, dists, label='error')
surffile = '.'.join(vtkfile.split('.')[:-1]) + '_surf.vtk'
sf.vtk.write_triangular_mesh(surffile, points, data.tris)
print("Wrote:", vtkfile)
