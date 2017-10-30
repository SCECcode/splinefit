"""
Parser for the gmsh msh file format

"""
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
        out.append([int(data[0])] + map(lambda x : float(x), data[1:]))

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
        out.append(map(lambda x : int(x), data))

    out = np.array(out)
    return out




