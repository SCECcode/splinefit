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
    corners = list(corner_ids) + [corner_ids[0]]
    boundaries = []
    for i, ci in enumerate(corners[:-1]):
        data, ids = get_segment(points, corners[i], corners[i+1])
        boundary = helper.Struct(
                      {'points': data,
                           'x' : data[:,0], 
                           'y' : data[:,1], 
                           'z' : data[:,2],
                           'ids' : ids})
        boundaries.append(boundary)
    return boundaries

def make_plot(data):
    helper.plot_points2(data.bnd_rxy,'bo')
    helper.plot_points2(data.corners,'k*')
    helper.plot_points2(data.bottom, 'C0-')
    helper.plot_points2(data.right, 'C1-')
    helper.plot_points2(data.top, 'C2-')
    helper.plot_points2(data.left, 'C3-')

    if figfile:
        plt.savefig(figfile)

def plot_boundaries(corners, points, boundaries, figfile):
    helper.plot_points2(corners,'ko')
    for bnd in boundaries:
        ids = bnd['ids']
        plt.plot(points[ids, 0], points[ids, 1])

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
boundaries = edges(points, corner_ids)
data.corners = data.bnd_rxy[corner_ids]
data.corner_ids = corner_ids
data.boundaries = boundaries

plot_boundaries(data.corners, pts, boundaries, '')
helper.show(showplot)

pickle.dump(data, open(outputfile, 'wb'))
