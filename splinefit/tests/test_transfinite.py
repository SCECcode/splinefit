import numpy as np
import splinefit as sf
import pytest

def boundaries():
    # Setup a simple boundaries that are correctly defined
    xl = np.array([0.0, 0.0, 0.0])
    yl = np.array([0.0, 0.5, 1.0])
    xr = np.array([1.0, 1.0, 1.0])
    yr = np.array([0.0, 0.5, 1.0])
    xb = np.array([0.0, 0.5, 1.0])
    yb = np.array([0.0, 0.0, 0.0])
    xt = np.array([0.0, 0.5, 1.0])
    yt = np.array([1.0, 1.0, 1.0])
    return xl, yl, xr, yr, xb, yb, xt, yt

def wavy_boundary(nu=50, nv=50, A=0.2):

    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)

    xl = A*np.sin(2*np.pi*v)
    yl = v

    xr = 0*v + 1
    yr = v

    xb = u 
    yb = 0*u 

    xt = u 
    yt = 1 + 0*u 

    return xl, yl, xr, yr, xb, yb, xt, yt

def test_checkcorners():
    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    assert sf.transfinite.checkcorners(xl, yl, xr, yr, xb, yb, xt, yt)
    xl[0] = 0.1
    with pytest.warns(UserWarning) : status = sf.transfinite.checkcorners(xl, yl,
                                            xr, yr, xb, yb, xt, yt)

    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    yl[0] = 0.1
    with pytest.warns(UserWarning) : status = sf.transfinite.checkcorners(xl, yl,
                                            xr, yr, xb, yb, xt, yt)

    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    yr[0] = 0.1
    with pytest.warns(UserWarning) : status = sf.transfinite.checkcorners(xl, yl,
                                            xr, yr, xb, yb, xt, yt)

def test_checkboundaries():
    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    status = sf.transfinite.checkboundaries(xl, yl, xr, yr, xb, yb, xt, yt)
    assert status == 1

    yl = yl[::-1]
    with pytest.warns(UserWarning) : status = sf.transfinite.checkboundaries(xl, yl,
                                            xr, yr, xb, yb, xt, yt)
    assert status == 0

def test_fixboundaries():
    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    yl[1] = -0.2
    with pytest.warns(UserWarning) : status = sf.transfinite.fixboundaries(xl, yl,
                                            xr, yr, xb, yb, xt, yt)[0]
    assert status == 0

def test_fixcorners():
    xl, yl, xr, yr, xb, yb, xt, yt = boundaries()
    xl2, yl2, xr2, yr2, xb2, yb2, xt2, yt2 = sf.transfinite.fixcorners(xl, yl,
                                                                       xr, yr, 
                                                                       xb, yb, 
                                                                       xt, yt)
    assert np.all(np.isclose(xl2,xl))
    assert np.all(np.isclose(yl2,yl))
    assert np.all(np.isclose(xr2,xr))
    assert np.all(np.isclose(yr2,yr))
    assert np.all(np.isclose(xb2,xb))
    assert np.all(np.isclose(yb2,yb))
    assert np.all(np.isclose(xt2,xt))
    assert np.all(np.isclose(yt2,yt))

def test_bilinearinterp():
    import matplotlib.pyplot as plt
    nu = 10
    nv = 40
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)

    # Check that the unit square is recovered
    xl, yl, xr, yr, xb, yb, xt, yt = wavy_boundary(nu, nv, A=0)
    assert sf.transfinite.checkcorners(xl, yl, xr, yr, xb, yb, xt, yt)
    assert sf.transfinite.checkboundaries(xl, yl, xr, yr, xb, yb, xt, yt)
    X, Y = sf.transfinite.bilinearinterp(xl, yl, xr, yr, xt, yt, xb, yb, U, V)
    assert np.all(np.isclose(X,U))
    assert np.all(np.isclose(Y,V))

    # Do visual inspection of sin function applied to the left boundary 
    xl, yl, xr, yr, xb, yb, xt, yt = wavy_boundary(nu, nv, A=0.2)
    assert sf.transfinite.checkcorners(xl, yl, xr, yr, xb, yb, xt, yt)
    assert sf.transfinite.checkboundaries(xl, yl, xr, yr, xb, yb, xt, yt)
    X, Y = sf.transfinite.bilinearinterp(xl, yl, xr, yr, xt, yt, xb, yb, U, V)
    plt.plot(X, Y,'bo')
    plt.show()

