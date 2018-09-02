"""
Module for querying and manipulating triangulations

"""
import numpy as np

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

    edge_id = 0

    for tri_id, tri in enumerate(tris):
        tri_edges = tri_to_edges(tri)
        for edge_i in tri_edges:
            key = edge_mapping(*edge_i)
            # Insert new edge
            if key not in edges:
                edges[key] = {'id': edge_id, 'orientation' : [], 
                              'triangles' : []}
                edges[key]['orientation'].append(edge_reorder(edge_i) == edge_i)
                edges[key]['triangles'].append(tri_id)
                edge_id += 1
            else:
                # Key already exists
                edges[key]['orientation'].append(edge_reorder(edge_i) == edge_i)
                edges[key]['triangles'].append(tri_id)
    return edges


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
    import numpy as np
    id1 = edge[0]
    id2 = edge[1]

    assert is_node(id1) and is_node(id2)
    assert id1 != id2
    min_id = min(id1, id2)
    max_id = max(id1, id2)
    return (min_id, max_id)

def edge_mapping(id1, id2):
    """
    Define the mapping function for the edge-based data structure.

    """
    return "%s-%s" % edge_reorder((id1, id2))


def is_node(idx):
    """
    Check if the given index satisfies the constraints that define a node.
    """
    correct_type = isinstance(idx, np.int64) or isinstance(idx, np.int) \
           or isinstance(idx, int) 
    try:
        nonnegative = idx >= 0
    except:
        return False
    return correct_type and nonnegative

