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
    w = 1
    h = 1
    deg = 2
    c = 0.02*w*h
    eps = 0.5*c
    p0 = parameteric_ellipse(w, h)
    pp = parameteric_ellipse(w+eps, h+eps)
    pm = parameteric_ellipse(w-eps, h-eps)
    M0 = sf.i3l.vander2d(p0, deg)
    Mp = sf.i3l.vander2d(pp, deg)
    Mm = sf.i3l.vander2d(pm, deg)
    coef = sf.i3l.solve(M0, Mp, Mm, c)[0]
    
    nx = 100
    ny = 100
    x = np.linspace(-1.0, 1, nx)
    y = np.linspace(-1.0, 1, ny)
    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    pts = np.stack((x_, y_)).T
    M0 = sf.i3l.vander2d(pts, deg)
    
    src = M0.dot(coef)
    levels=[1e-6]
    Z = src.reshape(nx, ny)
    import matplotlib.pyplot as plt
    #plt.plot(p0[:,0],p0[:,1],'b-')
    #plt.plot(pp[:,0],pp[:,1],'b-')
    #plt.plot(pm[:,0],pm[:,1],'b-')
    levels = [0.0]
    plt.contour(x, y, Z, levels)
    plt.show()

def parameteric_rectangle(w, h, npts=100):
    # w: width, h : height
    u = np.linspace(-2*np.pi, np.pi*2, npts)
    x = 0.5*w*np.sign(np.cos(u))
    y = 0.5*h*np.sign(np.sin(u))
    return np.stack((x, y)).T

def parameteric_ellipse(w, h, npts=100):
    # w: width, h : height
    u = np.linspace(-np.pi, np.pi, npts)
    x = w*np.cos(u)
    y = h*np.sin(u)
    return np.stack((x, y)).T

def parameteric_fun(w, h, npts=100):
    # w: width, h : height
    u = np.linspace(-np.pi, np.pi, npts)
    x = w*(1 + np.cos(u))*np.cos(u) - 0.5*w
    y = h*(1 + np.cos(u))*np.sin(u) - 0.5*h
    return np.stack((x, y)).T

def test_parameteric_rectangle():
    return
    import matplotlib.pyplot as plt
    pts = parameteric_rectangle(1,1)
    plt.plot(pts[:,0],pts[:,1])
    plt.show()
