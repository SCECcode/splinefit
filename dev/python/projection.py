import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import matplotlib.pyplot as plt
import pickle

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


data = pickle.load(open(inputfile, 'rb'))
pcl_xyz = data.coords[:,1:]
pcl_xyz, mu, std = sf.fitting.normalize(pcl_xyz)

edges = data.bnd_edges

bnd_xyz =pcl_xyz[edges[:,0],:]


basis = sf.fitting.pca(bnd_xyz, num_components=3)
proj_basis = sf.fitting.pca(bnd_xyz, num_components=2)
ax = helper.plot_points(bnd_xyz)
bnd_xy = sf.fitting.projection(bnd_xyz, proj_basis)
pcl_xy = sf.fitting.projection(pcl_xyz, proj_basis)
helper.plot_points(bnd_xyz, ax, 'k')
helper.plot_points(bnd_xy, ax, 'b-')
helper.plot_basis(basis, ax)
plt.savefig(figfile)
helper.show(showplot)
data.mu = mu
data.std = std
data.basis = basis
data.proj_basis = proj_basis
data.edges = edges
data.bnd_xyz = bnd_xyz
data.bnd_xy = bnd_xy
data.pcl_xyz = pcl_xyz
data.pcl_xy = pcl_xy
data.bnd_proj_xyz = data.basis.T.dot(data.bnd_xyz.T).T
data.pcl_proj_xyz = data.basis.T.dot(data.pcl_xyz.T).T
pickle.dump(data, open(outputfile, 'wb'))
