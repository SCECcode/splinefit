import pytest
import splinefit as sf
import numpy as np

def test_tris_to_edges():
    tris = np.array([[1,2,3],[2,3,4]]).astype(np.int64)
    edges = sf.triangulation.tris_to_edges(tris)
    # Only one of the edges is shared
    assert len(edges) == 5
    # Check that both triangles appear in the shared edge
    assert 0 in edges['2-3']['triangles']
    assert 1 in edges['2-3']['triangles']

def test_tri_to_edges():
    tri = [1,2,3]
    assert sf.triangulation.tri_to_edges(tri)[0] == (1,2)
    assert sf.triangulation.tri_to_edges(tri)[1] == (2,3)
    assert sf.triangulation.tri_to_edges(tri)[2] == (3,1)

    # Not a triangle
    with pytest.raises(Exception) : sf.triangulation.tri_to_edges([0,1])

def test_unordered_boundary_edges():
    nodes = np.array([[1,2], [2,3], [1,3], [3,4], [2,4]]).astype(np.int64)
    count = np.array([1, 1, 0, 0, 0]) 
    bnd_edges = sf.triangulation.unordered_boundary_edges(nodes, count)
    bnd_edges_ans = np.array([[1,2], [2,3]])
    assert np.all(np.equal(bnd_edges,bnd_edges_ans))

def test_ordered_boundary_edges():
    nodes = np.array([[2,6], [1,4], [6,4], [1,2]]).astype(np.int64)
    bnd_edges_ans = np.array([[2,6], [6,4], [4,1], [1,2]])
    nodes_to_edges = sf.triangulation.nodes_to_edges(nodes)
    edges_to_nodes = nodes
    bnd_edges = sf.triangulation.ordered_boundary_edges(edges_to_nodes,
            nodes_to_edges)
    assert np.all(np.equal(bnd_edges,bnd_edges_ans))

def test_edges_shared_tri_count():
    tri = [1,2,3]
    tris = np.array([[1,2,3],[2,3,4]]).astype(np.int64)
    count_ans = [1, 2, 1, 1, 1]
    edges = sf.triangulation.tris_to_edges(tris)
    count = sf.triangulation.edges_shared_tri_count(edges)
    assert np.all(np.equal(count,count_ans))

def test_edges_to_nodes():
    tri = [1,2,3]
    tris = np.array([[1,2,3],[2,3,4]]).astype(np.int64)
    nodes_ans = np.array([[1,2], [2,3], [1,3], [3,4], [2,4]]).astype(np.int64)
    edges = sf.triangulation.tris_to_edges(tris)
    nodes = sf.triangulation.edges_to_nodes(edges)
    assert np.all(np.equal(nodes,nodes_ans))

def test_nodes_to_edges():
    nodes = np.array([[1,2], [2,3], [1,3], [3,4], [2,4]]).astype(np.int64)
    edges = sf.triangulation.nodes_to_edges(nodes)
    assert edges[1] == [0,2]
    assert edges[2] == [0,1,4]
    assert edges[3] == [1,2,3]
    assert edges[4] == [3,4]

def test_edge_mapping():
    assert sf.triangulation.edge_mapping(2,1) == '1-2'
    assert sf.triangulation.edge_mapping(1,2) == '1-2'
    
    # Nodes must be int
    with pytest.raises(Exception) : sf.triangulation.edge_mapping('1',1)
    with pytest.raises(Exception) : sf.triangulation.edge_mapping(1,'1')

def test_edge_inverse_mapping():
    key = '1-2'
    assert sf.triangulation.edge_inverse_mapping(key, True) == (1,2)
    assert sf.triangulation.edge_inverse_mapping(key, False) == (2,1)

def test_vectors():
    coords = np.array([[0.0, 1.0, 2.0],[1.0, -1.0, -2.0], [2.0, 3.0, 4.0]])
    edges = np.array([[0, 1], [1, 2]]).astype(np.int64)
    vectors = sf.triangulation.vectors(edges, coords)
    assert vectors.shape[0] == edges.shape[0]
    assert vectors.shape[1] == coords.shape[1]
    assert np.all(np.equal(vectors[0,:], coords[1,:] - coords[0,:]))

def test_edge_reorder():
    assert sf.triangulation.edge_reorder((2,1)) == (1,2)
    assert sf.triangulation.edge_reorder((1,2)) == (1,2)

def test_is_node():
    assert sf.triangulation.is_node(int(1))
    assert sf.triangulation.is_node(np.int(1))
    assert sf.triangulation.is_node(np.int64(1))
    assert not sf.triangulation.is_node(1.0)
    # Negative node ids not allowed
    assert not sf.triangulation.is_node(int(-1))

def test_is_edge():
    assert sf.triangulation.is_edge((int(1),int(10)))
    assert not sf.triangulation.is_edge((int(1),int(1)))
    assert not sf.triangulation.is_edge((int(1),int(1),int(2)))

def test_normal2():
    coords = np.array([[0.0, 0.0],[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    normals = sf.triangulation.normals2(coords)
    assert normals.shape[0] == coords.shape[0] - 1
    assert np.isclose(normals[0,0], 0)
    assert np.isclose(normals[0,1], -1)

def test_orientation2():
    coords = np.array([[0.0, 0.0],[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    normals = sf.triangulation.normals2(coords)
    is_ccw = sf.triangulation.orientation2(coords, normals)
    assert is_ccw > 0 
    coords = coords[::-1,:]
    normals = sf.triangulation.normals2(coords)
    is_ccw = sf.triangulation.orientation2(coords, normals)
    assert is_ccw < 0 


