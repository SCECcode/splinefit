#import bspline

def refine(m0, levels):
    return int((m0 + 1)*2**levels - 1)

def indices(level, m0, levels):
    r0 = refine(m0, level)
    rn = refine(m0, levels)
    stride = 2**(levels - level)
    return range(0, rn+2, stride)

def ctrl_indices(level, m0, levels):
    r0 = refine(m0, level)
    rn = refine(m0, levels)
    stride = 2**(levels - level -1)
    return range(0, rn-4, stride)

def knot_indices(idx, p):
    return [p + i for i in idx]
   

    #range(0, int(r0), 1+int(2**(level-1)))



