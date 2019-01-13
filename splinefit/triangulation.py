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
    import warnings

    edges = {}

    edge_id = 0

    for tri_id, tri in enumerate(tris):
        tri_edges = tri_to_edges(tri)
        for edge_i in tri_edges:
            if not is_edge(edge_i):
                warnings.warn('Found invalid edge: %s'%str(edge_i))
                continue
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
    return np.array(nodes[tri_count == boundary_count,:]).astype(np.int64)

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
    out = np.zeros((num_edges,2)).astype(np.int64)
    # Start ordering with some arbitrary edge
    node1 = edges_to_nodes[0][0]
    node2 = edges_to_nodes[0][1]
    visited = {}
    num_traversed = 0
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
        if node2 in visited:
            break
        else:
            visited[node2] = 1
        num_traversed += 1



    return out[:num_traversed,:]

def boundary_loops(edges_to_nodes, nodes_to_edges):
    """
    Order the edges on the boundary. The edges are listed so that the last node
    of the current edge is the first node of the next edge. This function does
    not take counter clockwise or clockwise traversal into account. The returned
    list of edges may be ordered in either direction.  

    The last node in the last edge will not get correctly updated if the
    boundary does not form a closed loop.

    This function will return an array that contains the columns:

    node ID 1, node ID 2, traversal ID, loop ID

    The first two columns define an edge, the second is the order in which this
    edge is visited for a given loop. The final index defines which loop a set
    of edges belong to.

    An edge can only belong to one loop. Meshes that have edges that share
    multiple loops are not supported.

    """
    import warnings
    num_edges = len(edges_to_nodes)
    out = np.zeros((num_edges,4)).astype(np.int64)
    # Start ordering with some arbitrary edge
    node1 = edges_to_nodes[0][0]
    node2 = edges_to_nodes[0][1]
    visited = {}
    num_traversed = 0
    # Find the next edge that connects to node2, but isn't node1
    current_edge = 0
    first = node1
    mask_id = 1
    traversal_id = 0

    for i in range(num_edges):
        if out[current_edge,3] == 0:
            out[current_edge,0] = node1
            out[current_edge,1] = node2
            out[current_edge,2] = traversal_id
            out[current_edge,3] = mask_id
            old_node1 = node1
            node1 = node2
        else:
            warnings.warn('The edge %d belongs to more than one loop.'
                          %current_edge)

        # Find next node2
        current_edge = -1
        for edge_id in nodes_to_edges[node1]:
            edge_j = edges_to_nodes[edge_id]
            if edge_j[0] != old_node1 and edge_j[0] != node2:
                node2 = edge_j[0]
                current_edge = edge_id
                break
            if edge_j[1] != old_node1 and edge_j[1] != node2:
                node2 = edge_j[1]
                current_edge = edge_id
                break

        # Found end of loop (back to beginning)
        if node2 == first:
            traversal_id += 1
            out[current_edge,0] = node1
            out[current_edge,1] = node2
            out[current_edge,2] = traversal_id
            out[current_edge,3] = mask_id

            # Reset search
            current_edge = -1
            for j in range(num_edges):
                if out[j,3] == 0:
                    current_edge = j
                    node1 = edges_to_nodes[j][0]
                    node2 = edges_to_nodes[j][1]
                    break

            # Exit if no remaining loops can be found
            if current_edge == -1:
                break

            traversal_id = 0
            first = node1
            mask_id += 1
        else:
            traversal_id += 1
            
        if node2 in visited:
            break
        else:
            visited[node2] = 1

    return out[:,:]

def get_loop(edges, loop_id):
    """
    Return an array of edges that belong to a loop. Use `boundary_loops` to
    find loops. Each row in the output contains the two node Ids, and the nodes
    are ordered in the direction of traversal.
    """

    # Get number of elements in loop
    num_nodes = sum(edges[:,3] == loop_id)

    out = np.zeros((num_nodes,2)).astype(np.int64)

    for i in range(edges.shape[0]):
        edge = edges[i,:]
        if edge[3] == loop_id:
            traversal = edge[2] 
            node1 = edge[0]
            node2 = edge[1]
            out[traversal,0] = node1
            out[traversal,1] = node2

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

def vectors(edges, coords):
    """
    Convert edges to vectors.

    Arguments:
        edges : An np.array of size m x n, where m is the number of edges and
            `n=2` is the node ids.
        coords : An np.array of size p x q, where q is the number of nodes and
            `q=3` is their coordinates in 3D space.

    """
    
    pts1 = coords[edges[:,0],:]
    pts2 = coords[edges[:,1],:]
    return pts2 - pts1

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

def close_boundary(points):
    """
    Close the boundary by adding the first point to the end of the array.

    Arguments:
        points : Array of points (size: num points x 3).

    """
    return np.vstack((points, points[0,:]))

def is_closed(edges):
    """
    Check if the boundary is closed by observing if the first edge is the same
    as the last edge
    """
    first = edges[0,:]
    last  = edges[edges.shape[0]-1,:]

    if first[1] == last[0] and first[0] == last[1]:
        return True

    return False


def normals2(points):
    """
    Compute normals along a boundary segment. Determines the normals for either
    an open or closed boundary (use `close_boundary` to close a boundary
    segment). 

    The boundary points must be ordered. The outward pointing normal is returned
    if the boundary points are ordered counter-clockwise.

    Arguments:
        points : Array of points (size: num points x 2).

    Returns:
        out : Normal at edge. The edges are defined as e_i = (p_{i+1}, p_i) 

    """
    ex = points[0:-1,0] - points[1:,0]
    ey = points[0:-1,1] - points[1:,1]

    nx = -ey
    ny = ex
    n = np.vstack((nx,ny)).T
    return n

def orientation2(points, normals):
    """
    Determine the orientation of an ordered list of points along a boundary
    segment. 

    Arguments:
        points : Array of points (size: num points x 2).
        normals : Array of normal components (size num points - 1 x 2).

    Returns:
        out. Boundary orientation. `out > 0` if the boundary points are ordered
        counter-clockwise.

    """
    mx = points[0:-1,0] + points[1:,0]
    my = points[0:-1,1] + points[1:,1]

    return(sum(mx*normals[:,0] + my*normals[:,1]))

def circumference(points, edges):
    """
    Estimate the area of a loop given the coordinates and edges of the loop.

    points : Is a global array of coordinates (x, y, z)

    """
    local_points = points[edges[:,0], :]
    local_points = close_boundary(local_points)
    normals = normals2(local_points)
    lengths = np.linalg.norm(normals, axis=1)
    return sum(lengths)
