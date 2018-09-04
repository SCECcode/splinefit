import sys
from splinefit import msh
import splinefit as sf
import numpy as np

inputfile = sys.argv[1]
outputfile = sys.argv[2]
if len(sys.argv) < 4:
    figfile = None
else:
    figfile = sys.argv[3]

def get_boundary(tris):
    
    # Extract triangles from gmsh data and shift to zero indexing
    tris = msh.get_data(tris, num_members=3)
    
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
    return bnd_edges

def segment_boundary(coords, bnd_edges, num_corners=4):
    vectors = sf.triangulation.vectors(bnd_edges, coords)
    #FIXME: Boundary segmentation should probably not take place until the data
    # has been projected onto the best fitting plane.
    # As a quick hack, just use y, z coordinates (this fails for some meshes)
    width=2
    vectors = coords[bnd_edges[:-width,0],:] - coords[bnd_edges[width:,0],:]
    v1 = vectors[0:-1,1:]
    v2 = vectors[1:,1:]
    num_nodes = v1.shape[0]

    for i in range(num_nodes):
        norm_v1 = np.sqrt(v1[i,:].dot(v1[i,:]))
        norm_v2 = np.sqrt(v2[i,:].dot(v2[i,:]))
        v1[i,:] = v1[i,:]/norm_v1
        v2[i,:] = v2[i,:]/norm_v2

    dots = np.zeros((num_nodes,))

    for i in range(num_nodes):
        norm_v1 = np.sqrt(v1[i,:].dot(v1[i,:]))
        norm_v2 = np.sqrt(v2[i,:].dot(v2[i,:]))
        dots[i] = np.abs(v1[i,:].dot(v2[i,:])/(norm_v1*norm_v2))

    dots_idx = np.argsort(dots)

    dots_idx = dots_idx[0:num_corners]
    #dots_idx = dots_idx[3:4]
    seg1 = bnd_edges[dots_idx,1] - bnd_edges[dots_idx,0]
    seg2 = bnd_edges[dots_idx,1] - bnd_edges[dots_idx+1,0]

    print(v1[dots_idx,:])
    print(v2[dots_idx,:])
    #print(coords[seg1,:])
    #print(coords[seg2,:])

    #print(bnd_edges)

    #print(dots_idx)
    seg0 = bnd_edges[dots_idx,0]
    seg1 = bnd_edges[dots_idx,1]
    seg2 = bnd_edges[dots_idx+1,1]
    #print(seg0, seg1, seg2)
    #print(bnd_edges[dots_idx,:])
    #print(bnd_edges[dots_idx+1,:])

    w1 = coords[seg1,1:] - coords[seg0,1:]
    w2 = coords[seg2,1:] - coords[seg1,1:]
    w1 = w1[0]
    w2 = w2[0]

    #print(coords[seg0,:])
    #print(coords[seg1,:])
    #print(coords[seg2,:])

    print(w1)
    print(w2)

    norm_w1 = np.sqrt(w1.dot(w1.T))
    norm_w2 = np.sqrt(w2.dot(w2.T))

    uw1 = w1/norm_w1
    uw2 = w2/norm_w2
    
    print(w1/norm_w1)
    print(w2/norm_w2)
    print(uw1.dot(uw2.T))

    pt_ids0 = bnd_edges[:,0]
    pt_ids1 = bnd_edges[:,1]
    bnd_coords = coords[pt_ids0,:]
    bnd_coords2 = coords[pt_ids1,:]
    bnd_coords = np.vstack((bnd_coords, bnd_coords2[-1,:]))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(bnd_coords[:,1], bnd_coords[:,2], bnd_coords[:,3],'ko--')
    ax.plot(coords[seg1,1], coords[seg1,2], coords[seg1,3],'r*')
    #print(w2)
    #ax.plot(coords[seg1,1] + w1[0], coords[seg1,2] + w1[1], coords[seg1,3] +
    #        w1[2],'m*')
    #ax.plot(coords[seg1,1] + w2[0], coords[seg1,2] + w2[1], coords[seg1,3] +
    #        w2[2],'g*')
    #ax.plot(coords[seg2,1], coords[seg2,2], coords[seg2,3],'b*')
    #hull = ConvexHull(coords)
    #for simplex in hull.simplices:
    #    ax.plot(coords[simplex, 1], coords[simplex, 2], coords[simplex, 3], 'k-')
    plt.show()
    return seg1

def make_plot(coords, tris, edges, figfile):
    if not figfile:
        return
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    pt_ids0 = bnd_edges[:,0]
    pt_ids1 = bnd_edges[:,1]
    bnd_coords = coords[pt_ids0,:]
    bnd_coords2 = coords[pt_ids1,:]
    bnd_coords = np.vstack((bnd_coords, bnd_coords2[-1,:]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(coords[:,1], coords[:,2], coords[:,3], triangles=tris)
    ax.plot(bnd_coords[:,1], bnd_coords[:,2], bnd_coords[:,3],'k-')
    plt.savefig(figfile)
    print("Wrote figure:", figfile)

def export(coords, bnd_edges, outputfile):
    #TODO: Export to gmsh. This exporter is not yet working.
    num_edges = bnd_edges.shape[0]
    # id, elem type, num tags, tags, node list
    num_attr = 6
    elems = np.zeros((num_edges, num_attr)).astype(np.int64)
    for i in range(num_edges):
        elems[i,0] = i + 1
        elems[i,1] = 1
        elems[i,2] = 1
        elems[i,3] = 1
        elems[i,4] = bnd_edges[i,0] + 1
        elems[i,5] = bnd_edges[i,1] + 1
    sf.msh.write(outputfile, coords, elems)

coords, tris = sf.msh.read(inputfile)
bnd_edges = get_boundary(tris)
#segments = segment_boundary(coords, bnd_edges)
make_plot(coords, tris, bnd_edges, figfile)
export(coords, bnd_edges, outputfile)
