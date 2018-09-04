import sys
from splinefit import msh
import splinefit as sf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputfile = sys.argv[1]
outputfile = sys.argv[2]
if len(sys.argv) < 4:
    figfile = None
else:
    figfile = sys.argv[3]

coords, tris = sf.msh.read(inputfile)

edges = sf.msh.get_data(tris)

print(edges.shape)
print(edges)

bnd_coords = coords[edges[:,0],:]

xyz = bnd_coords[:,1:]

eig_vec = sf.fitting.pca(xyz)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xyz[:,0], xyz[:,1], xyz[:,2],'k')
plt.show()
