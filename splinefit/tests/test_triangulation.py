import pytest

def test_edge_mapping():
    import splinefit as sf

    assert sf.triangulation.edge_mapping(2,1) == '1-2'
    assert sf.triangulation.edge_mapping(2,1) == '1-2'

    # Nodes in a edge must be unique
    with pytest.raises(Exception) : sf.triangulation.edge_mapping(1,1)
    
    # Nodes must be int
    with pytest.raises(Exception) : sf.triangulation.edge_mapping('1',1)
    with pytest.raises(Exception) : sf.triangulation.edge_mapping(1,'1')
