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
        return np.vstack((points[id1:,:],points[:id2+1,:]))
    else:
        return points[id1:id2+1,:]

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
    bottom = get_segment(points, corner_ids[0], corner_ids[1])
    right = get_segment(points, corner_ids[1], corner_ids[2])
    top = get_segment(points, corner_ids[2], corner_ids[3])
    left = get_segment(points, corner_ids[3], corner_ids[0])
    return bottom, right, top, left

def make_plot(data):
    helper.plot_points2(data.rxy,'bo')
    helper.plot_points2(data.corners,'k*')
    helper.plot_points2(data.bottom, 'C0-')
    helper.plot_points2(data.right, 'C1-')
    helper.plot_points2(data.top, 'C2-')
    helper.plot_points2(data.left, 'C3-')

    if figfile:
        plt.savefig(figfile)
    plt.show()

data = pickle.load(open(inputfile, 'rb'))
data.rxy = fix_orientation(data.rxy)
bbox = sf.fitting.bbox2(data.rxy)
corner_ids = get_corners(data.rxy, bbox)
data.bottom, data.right, data.top, data.left = edges(data.rxy, corner_ids)
data.corners = data.rxy[corner_ids]

make_plot(data)

pickle.dump(data, open(outputfile, 'wb'))
