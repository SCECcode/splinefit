"""usage:
split_boundary.py <--inputfile=string> <--outputfile=string> <--max_angle=float> 

split_boundary.py (--help)

    Required arguments:

     inputfile:     Load pickle binary data file containing boundary data
    outputfile:     Write pickle binary data file
     max_angle:     Threshold criterion used to mark boundary points as
                    corner points

    Optional arguments:

        help: Display additional helpful information  

"""
import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt
import pickle
import scipy.optimize

helptxt = \
"""This tool splits the boundary into multiple segments. The break points that
define the start and end of each segment are determined by a criterion that is
applied to each boundary point. This criterion uses the angle between two
neighboring edges. A point is flagged as a corner point if its angle exceeds a
user-defined threshold. 

For example,

Suppose that the boundary is defined by the node indices:

    id =  0   1   2   3   4 ...

having angles:

    phi = 0.4 1.6 0.6 0.2 0.4 ...

The angle `1.6` associated with node N_1 and this angle is obtained by measuring
the angle between the vectors defined by the Edges N_1N_2 and N_0N_1. If the
condition `max_phi < 1.6` is true,  then this node will be flagged as a corner
point. 

"""

def get_options(argv):
    """
    Get command line arguments.
    """

    options = helper.Struct()
    if '--help' in argv:
        print(helptxt)
        exit()
    try:
        args = helper.get_args(argv)
        options.inputfile = args['--input']
        options.outputfile = args['--output']
        options.max_angle = float(args['--max_angle']) * np.pi / 180
        
        if '--showplot' in args:
            options.showplot = int(args['showplot'])
        else:
            options.showplot = 0
        if '--figfile' in args:
            options.figfile = args['figfile']
        else:
            options.figfile = 0
    except:
        print(__doc__)
        exit()
    return options

def get_segment(points, id1, id2):
    if id2 < id1:
        ids = list(range(id1, points.shape[0])) + list(range(id2+1))
        return np.vstack((points[id1:,:],points[:id2+1,:])),ids
    else:
        ids = list(range(id1, id2+1))
        return points[id1:id2+1,:], ids

def fix_orientation(points):
    """
    Make sure that the boundary is ordered counter clockwise
    """
    normals = sf.triangulation.normals2(points)
    is_ccw = sf.triangulation.orientation2(points, normals)
    if is_ccw < 0:
        points = points[::-1,:]
    return points

def edges(points, corner_ids):
    corners = list(corner_ids) + [corner_ids[0]]
    boundaries = []
    for i, ci in enumerate(corners[:-1]):
        data, ids = get_segment(points, corners[i], corners[i+1])
        boundaries.append({'data' : data, 'ids' : ids})
    return boundaries

def plot_boundaries(corners, points, boundaries, figfile):
    helper.plot_points2(corners,'ko')
    for bnd in boundaries:
        ids = bnd['ids']
        plt.plot(points[ids, 0], points[ids, 1])

    if figfile:
        plt.savefig(figfile)

options = get_options(sys.argv)
data = pickle.load(open(options.inputfile, 'rb'))
pts = np.vstack((data.bnd_rxy[:,0], data.bnd_rxy[:,1], data.bnd_rz)).T
pts = fix_orientation(pts)
data.bnd_rxy = pts[:,0:2]
data.bnd_rz = pts[:,2]
phi = sf.triangulation.angles(pts)
corner_ids = np.argwhere(phi > options.max_angle).flatten()
boundaries = edges(pts, corner_ids)
data.corners = data.bnd_rxy[corner_ids]
data.corner_ids = corner_ids

plot_boundaries(data.corners, pts, boundaries, options.figfile)
helper.show(options.showplot)

pickle.dump(data, open(options.outputfile, 'wb'))
