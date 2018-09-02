"""
Module for querying and manipulating triangulations

"""

def tris_to_edges(tris):
    """
    Construct an edge to element data structure for a given triangulation.
    The edge element data structure defines a mapping between an edge (N_1, N_2)
    and all the elements (triangles) that share this edge and their orientation
    with respect to the edge. The key that defines the edge is represented as a
    string "1-2" where the first number is the Node ID of the node of this edge
    with the least index. 
    
    Arguments:
        tris : Triangulation in the form of a m x 3 array.

    Returns:
        edges : The edge-to-element data structure as explained above.

    """

    edges = {}

    #for idx, tri in enumerate(tris):
    #    tri_edges = tri_to_edges(tri)
    #    for edge_i in tri_edges:
    #        key = edge_mapping(edge_i)
    #        if key not in tri_edges:
    #            edges[key] = {'orientation' : [], 'triangles' : []}
    #            edges[key]['orientation'].append(


def tri_to_edges(tri):
    """
    Return the edges in a triangle.

    Arguments:
        tri : Element defined by 3 nodes
    Returns:
        A list of three tuples defining the edges of the given triangle.
    """
    assert len(tri) == 3
    return [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]

def edge_reorder(edge):
    """
    Reorder the nodes in an edge so that the node with the least index appears
    first.
    """
    id1 = edge[0]
    id2 = edge[1]

    assert isinstance(id1, int)
    assert isinstance(id2, int)
    assert id1 != id2
    min_id = min(id1, id2)
    max_id = max(id1, id2)
    return (min_id, max_id)

def edge_mapping(id1, id2):
    """
    Define the mapping function for the edge-based data structure.

    """
    return "%s-%s" % edge_reorder((id1, id2))

