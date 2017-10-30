"""
Parser for the gmsh msh file format

"""
_nodes = '\$Nodes\n(\d+)\n([\w\W]+)\n\$EndNodes'
_elements = '\$Elements\n(\d+)\n([\w\W]+)\n\$EndElements'

def read(filename):
    txt = open(filename, 'r').read()

    n = nodes(txt)
    e = elements(txt)

    return n, e

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
    
    return np.array(out)

