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

def get_boundary(coords, tris):
    # Extract triangles from gmsh data and shift to zero indexing
    
    # Extract all edges
    edges = sf.triangulation.tris_to_edges(tris)
    edges_to_nodes = sf.triangulation.edges_to_nodes(edges)
    print("Total number of edges:", edges_to_nodes.shape[0])
    
    # Extract all boundary edges (unordered)
    count = sf.triangulation.edges_shared_tri_count(edges)
    bnd_edges = sf.triangulation.unordered_boundary_edges(edges_to_nodes, count,
            boundary_count=1)
    print("Total number of boundary edges:", bnd_edges.shape[0])
    
    # Order boundary edges so that boundary can be easily traversed
    nodes_to_edges = sf.triangulation.nodes_to_edges(bnd_edges)
    bnd_edges = sf.triangulation.boundary_loops(bnd_edges,nodes_to_edges)
    num_loops = max(bnd_edges[:,3])
    circ = []
    print("Number of boundary loops:", num_loops)
    for loop_id in range(1, num_loops+1):
        loop = sf.triangulation.get_loop(bnd_edges, loop_id)
        c = sf.triangulation.circumference(coords, loop)
        local_points = coords[bnd_edges[:,0], :]
        local_points = sf.triangulation.close_boundary(local_points)
        normals = sf.triangulation.normals2(local_points) 
        print("Loop ID: %d, Number of boundary edges: %d Circumference: %g "\
               %(
                loop_id,
                sum(bnd_edges[:,3] == loop_id),
                c))
        circ.append(c)
    c_idx = np.argmax(circ)
    bnd_edges = sf.triangulation.get_loop(bnd_edges, c_idx + 1)
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

bnd_edges = get_boundary(coords, tris)
make_plot(coords, tris, bnd_edges, figfile)

data = helper.Struct()
data.coords = coords
data.tris = tris
data.bnd_edges = bnd_edges

pickle.dump(data, open(outputfile, 'wb'))
