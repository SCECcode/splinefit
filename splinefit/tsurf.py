"""
Parser for the GOCAD Tsurf file format.
Only vertex and element information is considered.
Any other data present in the Tsurf file format is currently ignored.
"""
_version = 'GOCAD TSurf (\d+)'
_header = 'HEADER \{([\w\W]+?)\}'
_vrtx = 'VRTX (\d+) ([-\w\.]+) ([-\w\.]+) ([-\w\.]+)'
_tri = 'TRGL (\d+) (\d+) (\d+)'
_surf = 'TFACE'

def read(filename, min_elems=0):
    txt = open(filename).read() 
    # Extract all triangular surfaces and process them one by one.
    surfs = txt.split(_surf)[1:]

    p = []
    t = []

    for surf in surfs:
        ti = tri(surf)
        pi = vrtx(surf)
        pi, ti = swap(pi, ti)
        if ti.shape[0] < min_elems:
            continue
        p.append(pi)
        t.append(ti)


    return p, t

def swap(p, t):
    """
    Replace node ID by the order in which it occurs in the node array "pt".

    """
    import numpy as np
    new_tri = np.copy(t)
    new_p = np.copy(p)

    for i in range(p.shape[0]):
        new_p[i,0] = i
        old_i = p[i,0]
        for k in range(3):
            ids = t[:,k] == old_i
            new_tri[ids,k] = i

    return new_p, new_tri

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
        out.append([int(mi) for mi in m])

    return np.array(out)

def msh(tri):
    import numpy as np

    n = tri.shape[0]
    out = np.zeros((n, 8))

    for i in range(n):
        out[i,0:8] = [i+1] + [2, 2, 0, 4] + list(tri[i,:])

    return out




