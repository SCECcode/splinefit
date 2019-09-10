import pytest
import splinefit as sf
import numpy as np

nodes = np.array([[2,6], [1,4], [6,4], [1,2], [3,7], [7,8], [8,3]]).astype(np.int64)
bnd_edges_ans = np.array([[2,6,0,1], [4,1,2,1], [6,4,1,1], [1,2,3,1], [3,7,0,2],
                          [7,8,1,2], [8,3,2,2]])
nodes_to_edges = sf.triangulation.nodes_to_edges(nodes)
edges_to_nodes = nodes
bnd_edges = sf.triangulation.boundary_loops(edges_to_nodes,
        nodes_to_edges)
print(bnd_edges)
loop = sf.triangulation.get_loop(bnd_edges,1)
loop = np.array([[0,1],[1,2],[2,3],[3,4]])
coords = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
print(loop)
c = sf.triangulation.circumference(coords, loop)
print(c)

