import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
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

def check_num_tris(tris, min_elem=16):
    # Make sure that there are a sufficient number of elements to treat

    if tris.shape[0] <= min_elem:
        print("Not enough elements! Boundary extraction aborted.")
        exit(0)

def get_boundary(tris):
    # Extract triangles from gmsh data and shift to zero indexing
    
    # Extract all edges
    edges = sf.triangulation.tris_to_edges(tris)
    edges_to_nodes = sf.triangulation.edges_to_nodes(edges)
    print("Total number of edges:", edges_to_nodes.shape[0])
    
    # Extract all boundary edges (unordered)
    count = sf.triangulation.edges_shared_tri_count(edges)
    bnd_edges = sf.triangulation.unordered_boundary_edges(edges_to_nodes, count,
            boundary_count=1)
    print("Number of boundary edges:", bnd_edges.shape[0])
    
    # Order boundary edges so that boundary can be easily traversed
    nodes_to_edges = sf.triangulation.nodes_to_edges(bnd_edges)
    bnd_edges = sf.triangulation.ordered_boundary_edges(bnd_edges,nodes_to_edges)
    print("Number of boundary edges:", bnd_edges.shape[0])
    return bnd_edges

def make_plot(coords, tris, edges, figfile):
    if not figfile:
        return
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xyz, mu, std = sf.fitting.normalize(coords[:,1:]) 
    
    pt_ids = bnd_edges[:,0]
    bnd_coords = xyz[pt_ids,:]
    bnd_coords = helper.close_boundary(bnd_coords)

    fig, ax = helper.plot_mesh(xyz, tris)
    helper.plot_points(bnd_coords, ax=ax, style='k-')
    plt.savefig(figfile)
    helper.show(showplot)

coords, tris = sf.msh.read(inputfile)
tris = msh.get_data(tris, num_members=3, index=1)
check_num_tris(tris)
bnd_edges = get_boundary(tris)
make_plot(coords, tris, bnd_edges, figfile)

data = helper.Struct()
data.coords = coords
data.tris = tris
data.bnd_edges = bnd_edges

pickle.dump(data, open(outputfile, 'wb'))