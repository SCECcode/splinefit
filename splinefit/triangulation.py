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

    If the edge in the data structure is "1-2" and belongs to some triangle with
    edge 1-2, then the orientation is said to be positive (True). On the other
    hand, if the edge in the triangle is 2-1 (reversed order) then the
    orientation is negative (False).
    
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

def edges_shared_tri_count(edges):
    """
    Return an array containing the number of triangles each edge belongs to.

    Arguments:
        edges : Edge data structure.

    """
    num_edges = len(edges)
    count = np.zeros((num_edges,))
    for key in edges:
        edge = edges[key]
        count[edge['id']] = len(edge['triangles'])
    return count

def edges_to_nodes(edges):
    """
    Return an array containing the nodes of each edge.

    Arguments:
        edges : Edge data structure.

    """
    num_edges = len(edges)
    nodes = np.zeros((num_edges,2)).astype(np.int64)
    for i, key in enumerate(edges):
        ids = edge_inverse_mapping(key)
        nodes[i,0] = ids[0]
        nodes[i,1] = ids[1]
    return nodes

def unordered_boundary_edges(nodes, tri_count, boundary_count=1):
    """
    Return an array containing the edges that lie on the boundary. No particular
    ordering of the boundary edges is attempted. 

    Arguments:
        nodes : Array of nodes in each edge ( num edges x 2).
        tri_count : Number of shared triangles for each edge.
        boundary_count(optional) : determines the count an edge must have to be
        classified as a boundary edge.

    """
    return np.array(nodes[tri_count == boundary_count,:])

def ordered_boundary_edges(edges_to_nodes, nodes_to_edges):
    """
    Order the edges on the boundary. The edges are listed so that the last node
    of the current edge is the first node of the next edge. This function does
    not take counter clockwise or clockwise traversal into account. The returned
    list of edges may be ordered in either direction.  

    The last node in the last edge will not get correctly updated if the
    boundary does not form a closed loop.

    """
    num_edges = len(edges_to_nodes)
    out = np.zeros((num_edges,2))
    # Start ordering with some arbitrary edge
    node1 = edges_to_nodes[0][0]
    node2 = edges_to_nodes[0][1]
    # Find the next edge that connects to node2, but isn't node1
    for i in range(num_edges):
        out[i,0] = node1
        out[i,1] = node2
        old_node1 = node1
        node1 = node2
        # Find next node2
        for edge_id in nodes_to_edges[node1]:
            edge_j = edges_to_nodes[edge_id]
            if edge_j[0] != old_node1 and edge_j[0] != node2:
                node2 = edge_j[0]
                break
            if edge_j[1] != old_node1 and edge_j[1] != node2:
                node2 = edge_j[1]
                break

    return out

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

def nodes_to_edges(nodes):
    """
    Returns a list of lists that contains the edges each node belongs to.

    Arguments:
        nodes : Array of nodes in each edge ( num edges x 2).

    Returns:
        out : A list of list as explained above. Also, see the notes below.
    
    Notes:

    The output `out[i]` is a list containing the ids of all the edges that have
    node `i` in common.

    """

    out = {}

    for i, nodes_i in enumerate(nodes):
        if nodes_i[0] in out:
            out[nodes_i[0]].append(i)
        else:
            out[nodes_i[0]] = [i]
        if nodes_i[1] in out:
            out[nodes_i[1]].append(i)
        else:
            out[nodes_i[1]] = [i]
    return out

def edge_reorder(edge):
    """
    Reorder the nodes in an edge so that the node with the least index appears
    first.
    """
    import numpy as np
    assert is_edge(edge)
    min_id = min(edge[0], edge[1])
    max_id = max(edge[0], edge[1])
    return (min_id, max_id)

def edge_mapping(id1, id2):
    """
    Define the mapping function for the edge-based data structure.

    """
    return "%s-%s" % edge_reorder((id1, id2))

def edge_inverse_mapping(key, orientation=True):
    """
    Reverse the edge mapping. That is, convert the string "1-2" to (1,2) or
    (2,1) depending on what the original orientation of the nodes in the edge
    was.

    Arguments:
        key : String defining the edge mapping to reverse.
        orientation : Orientation of nodes in the edge. Set to `False` if the
        ordering should be reversed.

    """
    ids = key.split('-')
    assert len(ids) == 2
    ids = (int(ids[0]), int(ids[1]))
    if orientation:
        return ids
    else:
        return (ids[1], ids[0])

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

def is_edge(edge):
    """
    Check that the input satsifies the constraints that the define an edge.
    """
    correct_len =  len(edge) == 2
    id1 = edge[0]
    id2 = edge[1]
    unique_ids = id1 != id2

    return is_node(id1) and is_node(id2) and unique_ids
