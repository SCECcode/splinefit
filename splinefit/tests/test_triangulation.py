import pytest
import splinefit as sf

def test_edge_mapping():
    assert sf.triangulation.edge_mapping(2,1) == '1-2'
    assert sf.triangulation.edge_mapping(1,2) == '1-2'

    # Nodes in an edge must be unique
    with pytest.raises(Exception) : sf.triangulation.edge_mapping(1,1)
    
    # Nodes must be int
    with pytest.raises(Exception) : sf.triangulation.edge_mapping('1',1)
    with pytest.raises(Exception) : sf.triangulation.edge_mapping(1,'1')

def test_edge_reorder():
    assert sf.triangulation.edge_reorder((2,1)) == (1,2)
    assert sf.triangulation.edge_reorder((1,2)) == (1,2)

def test_tri_to_edges():
    tri = [1,2,3]
    assert sf.triangulation.tri_to_edges(tri)[0] == (1,2)
    assert sf.triangulation.tri_to_edges(tri)[1] == (2,3)
    assert sf.triangulation.tri_to_edges(tri)[2] == (3,1)

    # Not a triangle
    with pytest.raises(Exception) : sf.triangulation.tri_to_edges([0,1])

