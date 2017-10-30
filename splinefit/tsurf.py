"""
Parser for the GOCAD Tsurf file format.
Only vertex and element information is considered.
Any other data present in the Tsurf file format is currently ignored.
"""
_version = 'GOCAD TSurf (\d+)'
_header = 'HEADER \{([\w\W]+?)\}'
_vrtx = 'VRTX (\d+) ([-\w\.]+) ([-\w\.]+) ([-\w\.]+)'
_tri = 'TRGL (\d+) (\d+) (\d+)'

def read(filename):
    txt = open(filename).read() 
    p = vrtx(txt)
    t = tri(txt)
    return p, t

def version(txt):
    import re
    p = re.compile(_version, re.M)
    match = p.findall(txt)

    if not match:
        raise Exception('Version number not found')

    return match[0]

def header(txt):
    import re
    p = re.compile(_header, re.M)
    match = p.findall(txt)

    if not match:
        raise Exception('Header not found')

    #TODO: Format header data
    return match

def vrtx(txt):
    import re
    import numpy as np
    p = re.compile(_vrtx, re.M)
    match = p.findall(txt)

    if not match:
        raise Exception('No vertices found.')

    fmt = lambda x: (int(x[0]), float(x[1]), float(x[2]), float(x[3]))

    out = []
    for m in match:
        out.append(fmt(m))

    return np.array(out)

def tri(txt):
    import re
    import numpy as np
    p = re.compile(_tri, re.M)
    match = p.findall(txt)

    if not match:
        raise Exception('No triangles found.')

    out = []
    for m in match:
        out.append(map(lambda x: int(x), m))

    return np.array(out)

def msh(tri):
    import numpy as np

    n = tri.shape[0]
    out = np.zeros((n, 8))

    for i in range(n):
        out[i,0:8] = [i+1] + [2, 2, 0, 4] + list(tri[i,:])

    return out




