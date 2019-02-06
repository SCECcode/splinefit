import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt
import pickle
import scipy.optimize
"""
This script splits the boundary up into four segments (left, bottom, right, top)
The basic principle behind the segmentation is to select corner points by using
the L1 distance.

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

def get_corners(points, bbox, norm=1):
    """
    Select boundary corners using the norm `norm`. Defaults to `1` (L1 norm).

    """
    nearest = []
    for i in range(4):
        nearest.append(sf.fitting.argnearest(points, bbox[i,:], ord=norm))
    return nearest

def get_segment(points, id1, id2):
    if id2 < id1:
        ids = list(range(id1, points.shape[0])) + list(range(id2+1))
        return np.vstack((points[id1:,:],points[:id2+1,:])),ids
    else:
        ids = list(range(id1, id2+1))
        return points[id1:id2+1,:], ids

def fix_orientation(points):
    """
    Make sure boundary is ordered counter clockwise
    """
    normals = sf.triangulation.normals2(points)
    is_ccw = sf.triangulation.orientation2(points, normals)
    if is_ccw < 0:
        points = points[::-1,:]
    return points

def edges(points, corner_ids):
    bottom, bottom_ids = get_segment(points, corner_ids[0], corner_ids[1])
    right, right_ids = get_segment(points, corner_ids[1], corner_ids[2])
    top, top_ids = get_segment(points, corner_ids[2], corner_ids[3])
    left, left_ids = get_segment(points, corner_ids[3], corner_ids[0])
    return bottom, right, top, left, bottom_ids, right_ids, top_ids, left_ids

def make_plot(data):
    helper.plot_points2(data.bnd_rxy,'bo')
    helper.plot_points2(data.corners,'k*')
    helper.plot_points2(data.bottom, 'C0-')
    helper.plot_points2(data.right, 'C1-')
    helper.plot_points2(data.top, 'C2-')
    helper.plot_points2(data.left, 'C3-')

    if figfile:
        plt.savefig(figfile)

data = pickle.load(open(inputfile, 'rb'))
pts = np.vstack((data.bnd_rxy[:,0], data.bnd_rxy[:,1], data.bnd_rz)).T
pts = fix_orientation(pts)
data.bnd_rxy = pts[:,0:2]
data.bnd_rz = pts[:,2]
bbox = sf.fitting.bbox2(data.bnd_rxy)
data.bbox = bbox
corner_ids = get_corners(data.bnd_rxy, bbox)
points = np.vstack((data.bnd_rxy[:,0], data.bnd_rxy[:,1], data.bnd_rz)).T
data.bottom, data.right, data.top, data.left, data.bottom_ids, data.right_ids,\
data.top_ids, data.left_ids = edges(points, corner_ids)
data.corners = data.bnd_rxy[corner_ids]
data.corner_ids = corner_ids

make_plot(data)
helper.show(showplot)

pickle.dump(data, open(outputfile, 'wb'))
