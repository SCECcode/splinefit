import numpy as np

def checkcorners(xl, yl, xr, yr, xb, yb, xt, yt, silent=False):
    """
    Check if the corner points of the boundary segments are overlapping or not.

    Arguments:
        xl, yl : x,y-coordinates for the left boundary
        xr, yr : x,y-coordinates for the right boundary
        xb, yb : x,y-coordinates for the bottom boundary
        xt, yt : x,y-coordinates for the top boundary
        silent : Disable warnings.

    Return value:
        `True` if the corners are overlapping, and `False` otherwise.

    """

    ok = 1

    def warn(message):
        import warnings
        if not silent:
            warnings.warn(message)

    def test_corner(p1x, p1y, p2x, p2y, corner, ok):
        test = np.isclose(p1x, p2x) and np.isclose(p1y, p2y)
        if not test:
            warn("Non-overlapping vertices in corner `%s` found." % corner)
            ok = 0

        return ok

    ok = ok and test_corner(xl[0], yl[0], xb[0], yb[0], 'bottom-left', ok) 
    ok = ok and test_corner(xr[0], yr[0], xb[-1], yb[-1], 'bottom-right', ok) 
    ok = ok and test_corner(xl[-1], yl[-1], xt[0], yt[0], 'top-left', ok) 
    ok = ok and test_corner(xr[-1], yr[-1], xt[-1], yt[-1], 'top-right', ok) 
    return ok

def checkboundaries(xl, yl, xr, yr, xb, yb, xt, yt, silent=False):
    """
    This function checks for correctly oriented boundaries. Any issues
    discovered are reported as warnings and some of them can be addressed if
    `fix = True`. 

    By convention, correct boundary orientation means that the grid points on
    the boundary are ordered in the direction of positive x-values, and
    y-values.  For example, the y-coordinates along the left boundary should be
    increasing, `yl[0] < yl[1] < ... `

    Return value:
        `True` if the boundaries are correctly oriented, and `False` otherwise.

    """
    def warn(message):
        import warnings
        if not silent:
            warnings.warn(message)


    assert len(xl) == len(yl) and len(xr) == len(yr) and len(xl) == len(xr)
    
    ok = 1

    bnd_data = {}
    bnd_data['left'] = yl
    bnd_data['right'] = yr
    bnd_data['bottom'] = xb
    bnd_data['top'] = xt

    test_orientation = lambda bnd : np.all(np.argsort(bnd) == \
                       list(range(len(bnd))))

    for side in bnd_data:
        if not test_orientation(bnd_data[side]):
            warn("""Inconsistent orientation for boundary `%s`. """ % side)
            ok = 0
    return ok

def fixboundaries(xl, yl, xr, yr, xb, yb, xt, yt, silent=False):
    """
    This function attempts to fix any incorrect boundary orientations due to
    listing points in the reverse order. Other issues, such as non-monotonically
    increasing values are not fixed.

    Arguments:
        xl, yl : x,y-coordinates for the left boundary
        xr, yr : x,y-coordinates for the right boundary
        xb, yb : x,y-coordinates for the bottom boundary
        xt, yt : x,y-coordinates for the top boundary
        silent : Disable warnings

    Return values:
        ok : `True` if the orientations could be fixed, or no fix was needed,
            and otherwise `False.`
        bnd : Tuple containing updated boundary values, ordered as the input
            arguments (`xl, yl, xr, ...`).

    """
    test_orientation = lambda bnd : np.all(np.argsort(bnd) == \
                                    list(range(len(bnd))))
    fix_orientation = lambda bnd : bnd[::-1]

    def warn(message):
        import warnings
        if not silent:
            warnings.warn(message)

    bnd_data = {}
    bnd_data['left'] = [yl, xl]
    bnd_data['right'] = [yr, xr]
    bnd_data['bottom'] = [xb, yb]
    bnd_data['top'] = [xt, yt]

    ok = 1
    for side in bnd_data:
        if not test_orientation(bnd_data[side][0]):
            bnd_data[side][0] = fix_orientation(bnd_data[side][0])
            bnd_data[side][1] = fix_orientation(bnd_data[side][1])
            # Retest after attempting to fix
            if not test_orientation(bnd_data[side][0]):
                warn("Unable to fix orientation for boundary `%s`." % side)
                ok = 0

    yl = bnd_data['left'][0]
    yr = bnd_data['right'][0]
    xb = bnd_data['bottom'][0]
    xt = bnd_data['top'][0]
    xl = bnd_data['left'][1]
    xr = bnd_data['right'][1]
    yb = bnd_data['bottom'][1]
    yt = bnd_data['top'][1]

    return ok, (xl, yl, xr, yr, xb, yb, xt, yt)

def fixcorners(xl, yl, xr, yr, xb, yb, xt, yt):
    """
    This function removes any non-overlapping corners by assigning the average
    of two corners points that should overlap.

    Arguments:
        xl, yl : x,y-coordinates for the left boundary
        xr, yr : x,y-coordinates for the right boundary
        xb, yb : x,y-coordinates for the bottom boundary
        xt, yt : x,y-coordinates for the top boundary
        
    Return value:
        bnd : Tuple containing updated boundary values, ordered as the input
            arguments (`xl, yl, xr, ...`).

    """
    xlb = 0.5*(xl[0] + xb[0])
    ylb = 0.5*(yl[0] + yb[0])
    xlt = 0.5*(xl[-1] + xt[0])
    ylt = 0.5*(yl[-1] + yt[0])

    xrb = 0.5*(xr[0] + xb[-1])
    yrb = 0.5*(yr[0] + yb[-1])
    xrt = 0.5*(xr[-1] + xt[-1])
    yrt = 0.5*(yr[-1] + yt[-1])

    xl[0] = xlb
    yl[0] = ylb
    xl[-1] = xlt
    yl[-1] = ylt

    xr[0] = xrb
    yr[0] = yrb
    xr[-1] = xrt
    yr[-1] = yrt

    xb[0] = xlb
    yb[0] = ylb
    xb[-1] = xrb
    yb[-1] = yrb

    xt[0] = xlt
    yt[0] = ylt
    xt[-1] = xrt
    yt[-1] = yrt

    return xl, yl, xr, yr, xb, yb, xt, yt

def bilinearinterp(xl, yl, xr, yr, xb, yb, xt, yt, U, V):
    """
    Generates a 2D structured grid by performing linear transfinite
    interpolation. 

    Input arguments:
        xl, yl : x,y-coordinates for the left boundary
        xr, yr : x,y-coordinates for the right boundary
        xb, yb : x,y-coordinates for the bottom boundary
        xt, yt : x,y-coordinates for the top boundary

    Returns:

    x, y : np.array
           Arrays of size `n` containing the grid coordinates of the
           interpolation.

    References

    Chapter 3.4, 
    Thompson, J.F., Soni, B.K. and Weatherill, N.P. eds., 1998. Handbook of grid
    generation. CRC press.
    """

    nu = U.shape[0]
    nv = U.shape[1]

    tu = lambda x : np.tile(x, (nu, 1))
    tv = lambda x : np.tile(x, (nv, 1)).T

    xl2 = tv(xl)
    yl2 = tv(yl)
    xr2 = tv(xr)
    yr2 = tv(yr)
    xb2 = tu(xb)
    yb2 = tu(yb)
    xt2 = tu(xt)
    yt2 = tu(yt)

    # Boolean sum U x V = U + V - UV

    x =   (1 - U)*xl2  + U*xr2 + (1 - V)*xb2  + V*xt2 \
        - (1 - U)*(1 - V)*xb[0] - (1 - U)*V*xt[0] - U*(1 - V)*xb[-1] -U*V*xt[-1]
    y =   (1 - U)*yl2  + U*yr2 + (1 - V)*yb2  + V*yt2 \
        - (1 - U)*(1 - V)*yb[0] - (1 - U)*V*yt[0] - U*(1 - V)*yb[-1] -U*V*yt[-1]
    return (x, y)


