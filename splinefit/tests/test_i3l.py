import splinefit as sf
import numpy as np

def test_vander2d():
    # Check linear bivariate polynomial
    p = np.array([[0.0, 1.0]])
    deg = 1
    M = sf.i3l.vander2d(p,deg)
    assert M.shape[0] == 1
    assert M.shape[1] == 3
    assert M[0,0] == 1
    assert M[0,1] == 0
    assert M[0,2] == 1

    # Check quadratic bivariate polynomial
    p = np.array([[0.4, 1.0]])
    deg = 2
    M = sf.i3l.vander2d(p,deg)
    assert M.shape[0] == 1
    assert M.shape[1] == 6
    assert np.isclose(M[0,0],1   )  # 1
    assert np.isclose(M[0,1],0.4 )  # x
    assert np.isclose(M[0,2],0.16)  # x^2
    assert np.isclose(M[0,3],1   )  # y
    assert np.isclose(M[0,4],0.4 )  # x*y
    assert np.isclose(M[0,5],1   )  # y^2

def test_solve():
    w = 1.0
    h = 1.0
    deg = 4
    c = 0.1
    eps = 1e-1
    p0 = parameteric_rectangle(w, h)
    pp = parameteric_rectangle(w+eps, h+eps)
    pm = parameteric_rectangle(w-eps, h-eps)
    M0 = sf.i3l.vander2d(p0, deg)
    Mp = sf.i3l.vander2d(pp, deg)
    Mm = sf.i3l.vander2d(pm, deg)
    coef = sf.i3l.solve(M0, Mp, Mm, c)[0]
    
    x = np.linspace(-0.5, 0.5, 20)
    y = np.linspace(-0.5, 0.5, 20)
    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    pts = np.stack((x_, y_)).T
    M0 = sf.i3l.vander2d(pts, deg)
    
    src = M0.dot(coef)
    levels=[1e-2]
    Z = src.reshape(20, 20)
    import matplotlib.pyplot as plt
    plt.contour(x, y, Z, [levels])
    plt.show()
    assert 0

def parameteric_rectangle(w, h, npts=100):
    # w: width, h : height
    u = np.linspace(-2*np.pi, np.pi*2, npts)
    x = 0.5*w*np.sign(np.cos(u))
    y = 0.5*h*np.sign(np.sin(u))
    x = 0.5*w*np.cos(u)
    y = 0.5*h*np.sin(u)
    return np.stack((x, y)).T

def test_parameteric_rectangle():
    import matplotlib.pyplot as plt
    pts = parameteric_rectangle(1,1)
    plt.plot(pts[:,0],pts[:,1])
    plt.show()
