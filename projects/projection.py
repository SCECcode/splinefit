import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
outputfile = sys.argv[2]
if len(sys.argv) < 4:
    figfile = None
else:
    figfile = sys.argv[3]

coords, tris = sf.msh.read(inputfile)

edges = sf.msh.get_data(tris)

xyz = helper.close_boundary(coords[edges[:,0],1:])
xyz, mu, std = sf.fitting.normalize(xyz)


basis = sf.fitting.pca(xyz, num_components=2)
ax = helper.plot_points(xyz)
xy = sf.fitting.projection(xyz, basis)
helper.plot_points(xyz, ax, 'k')
helper.plot_points(xy, ax, 'b-')
helper.plot_basis(basis, ax)
plt.savefig(figfile)
