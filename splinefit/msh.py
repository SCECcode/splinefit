"""
Parser for the gmsh file format

"""
import numpy as np
_nodes = '\$Nodes\n(\d+)\n([\w\W]+)\n\$EndNodes'
_elements = '\$Elements\n(\d+)\n([\w\W]+)\n\$EndElements'
_meshformat = '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n'

def read(filename):
    txt = open(filename, 'r').read()

    n = nodes(txt)
    e = elements(txt)

    return n, e

def write(filename, n, e):
    f = open(filename, 'w')

    f.write(_meshformat)

    # Nodes
    f.write('$Nodes\n')
    f.write('%d\n'%n.shape[0])
    for i in range(n.shape[0]):
        f.write('%d %f %f %f\n'%(n[i,0], n[i,1], n[i,2], n[i,3]))
    f.write('$EndNodes\n')
    
    # Elements
    f.write('$Elements\n')
    f.write('%d\n'%e.shape[0])
    for i in range(e.shape[0]):
        fmt = ' '.join(['%d']*len(e[i])) + '\n'
        f.write(fmt % tuple(e[i]) )
    f.write('$EndElements\n')
    f.close()

def nodes(txt):
    import re
    import numpy as np

    p = re.compile(_nodes, re.M)
    matches = p.findall(txt)

    if not matches:
        raise Exception('No nodes found')

    numnodes = matches[0][0]
    nodes = matches[0][1].split('\n')
    out = []
    for line in nodes:
        data = line.split(' ')
        out.append([int(data[0])] + [float(x) for x in data[1:]])

    return np.array(out)

def elements(txt):
    import re
    import numpy as np

    p = re.compile(_elements, re.M)
    matches = p.findall(txt)

    if not matches:
        raise Exception('No nodes found')

    numnodes = matches[0][0]
    elem = matches[0][1].split('\n')
    out = []
    for line in elem:
        data = line.split(' ')
        out.append([np.int(x) for x in data])

    out = np.array(out)
    return out

def get_data(elem, num_members=2, index=0):
    """
    Return the data from a gmsh data structure

    Arguments:
        elem : Gmsh data structure.
        num_members (optional) : Number of data members to extract. Defaults to
            `2` (edge).
        index (optional) : Convert to zero indexing. Enabled by default.

    """

    data = []
    for ei in elem:
        num_fields = len(ei)
        data.append(ei[num_fields-num_members:])
    data = np.array(data)

    if index == 0:
        data = data - 1

    return data

def write_geo(filename, num_surfaces):
 out = []
 out += ['SetFactory("OpenCASCADE");']
 out += ['a() = ShapeFromFile("%s");' % (filename + ".igs")]
 surfaces = ' '.join(['{ Surface{%d}; Delete; }' % (i + 1) for i in
                      range(num_surfaces)])
 out += ["BooleanUnion" + surfaces]
 with open(filename + ".geo", 'w') as fh:
     fh.write('\n'.join(out))


