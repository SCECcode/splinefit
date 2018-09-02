"""
Module for querying and manipulating triangulations

"""

def tri_to_edge(tri):
    """
    Construct an edge to element data structure for a given triangulation.
    The edge element data structure defines a mapping between an edge (N_1, N_2)
    and all the elements (triangles) that share this edge and their orientation
    with respect to the edge. The key that defines the edge is represented as a
    string "1-2" where the first number is the Node ID of the node of this edge
    with the least index. 
    
    Arguments:
        tri : Triangular in the form of a m x 3 array.

    Returns:
        edges : The edge-to-element data structure as explained above.

    """

    edges = {}

def edge_mapping(id1, id2):
    """
    Define the mapping function for the edge-based data structure.

    """
    assert isinstance(id1, int)
    assert isinstance(id2, int)
    assert id1 != id2
    min_id = min(id1, id2)
    max_id = max(id1, id2)
    return "%s-%s" % (min_id, max_id)

