import splinefit as sf
import matplotlib.pyplot as plt
import numpy as np

def test_cubic():
    n = 20
    s = np.linspace(0,1,n)

    y = np.zeros((n,4))

    for i, si in enumerate(s):
        z = sf.bspline.cubic(si)
        for j in range(4):
            y[i,j] = z[j]

    plt.plot(s,y)
    #plt.show()

def test_curve():
    # Introduce one extra "ghost" control point at the end of the interval  
    P = np.array([1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0])
    x, y = sf.bspline.eval(P)
    plt.clf()
    plt.plot(x,y)
    #plt.show()

def test_normalize():
    P = np.array([1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0])
    n = 5
    nP, nc = sf.bspline.normalize(P, n)
    assert np.isclose(min(nP),0)
    assert np.isclose(max(nP),n)

    rP = sf.bspline.denormalize(nP, nc, n)
    assert np.all(np.isclose(P - rP,0))

def test_minimize():
    """
    Compare least square minimization procedure to scipy's 
    """
    return
    npts = 60
    px = np.linspace(-3+1e-2, 3-1e-2, npts)
    py = np.exp(-px**2) + 0.01 * np.random.randn(npts)
    from scipy.interpolate import make_lsq_spline, BSpline

    n = 9
    p = 3
    t = np.linspace(-2,2,20)
    U = np.r_[(px[0],)*(p+1),
              t,
              (px[-1],)*(p+1)]

    P = sf.bspline.bspline_lsq(px, py, U, p)

    u = np.linspace(-3,3,100)
    z = []
    for ui in u:
        z.append(sf.bspline.curvepoint(p, U, P, ui))
    spl = make_lsq_spline(px, py, U, p)
    plt.clf()
    plt.plot(px, py, 'bo')
    plt.plot(u, spl(u), 'g-', lw=3, label='LSQ spline')
    plt.plot(u, z,'k')

def test_bspline_curve():
    from scipy.interpolate import BSpline
    #return
    npts = 20
    a = -2
    b = 2
    p = 3
    m = 5 
    px = np.linspace(a, b, npts)
    py = np.sin(px)*np.exp(-px**2)# + 0.1 * np.random.randn(npts)
    px = np.cos(0.1*px)

    a = 0
    b = 1
    s = sf.bspline.chords(px, py, a=a, b=b)
    U = sf.bspline.kmeansknots(s, m, p, a=a, b=b)
    Px, Py, res = sf.bspline.lsq2(s, px, py, U, p)
    zx = []
    zy = []
    u = np.linspace(a,b,100)
    spl_x = BSpline(U, Px, p)
    spl_y = BSpline(U, Py, p)
    sx = []
    sy = []
    for ui in u:
        zx.append(sf.bspline.curvepoint(p, U, Px, ui))
        zy.append(sf.bspline.curvepoint(p, U, Py, ui))
        sx.append(spl_x(ui))
        sy.append(spl_y(ui))

    plt.clf()
    plt.plot(sx, sy,'k-')
    plt.plot(zx, zy,'b--')
    plt.plot(px, py,'k*')
    #plt.show()

def test_bspline_surface():
    return
    n1 = 16
    n2 = 16
    nu = 4
    nv = 4
    p = 3
    u = np.linspace(0, 1, n1)
    v = np.linspace(0, 1, n2)
    u, v = np.meshgrid(u, v)
    X = (0.2 + v + 1e-2*np.random.randn(n1,n2))*np.cos(u)
    Y = (0.2 + v + 1e-2*np.random.randn(n1,n2))*np.sin(u)
    plt.clf()
    plt.plot(X[:], Y[:], 'bo-')
    plt.plot(X.T[:], Y.T[:], 'bo-')

    u = u.flatten()
    v = v.flatten()
    x = X.flatten()
    y = Y.flatten()
    # Make sure that ordering if points doesn't matter
    idx = np.random.permutation(len(u))
    u = u[idx]
    v = v[idx]
    x = x[idx]
    y = y[idx]


    U = sf.bspline.uniformknots(nu, p)
    V = sf.bspline.uniformknots(nv, p)
    Px, res = sf.bspline.lsq2surf(u, v, x, U, V, p)
    Py, res = sf.bspline.lsq2surf(u, v, y, U, V, p)

    r = 3
    plt.plot(Px[:-r,:-r], Py[:-r,:-r], 'go-')
    plt.plot(Px.T[:-r,:-r], Py.T[:-r,:-r], 'go-')

    u = np.linspace(0, 1, 12)
    v = np.linspace(0, 1, 12)
    X = sf.bspline.evalsurface(p, U, V, Px, u, v)
    Y = sf.bspline.evalsurface(p, U, V, Py, u, v)
    plt.plot(X, Y, 'ro-')
    plt.plot(X.T, Y.T, 'ro-')
    plt.legend()
    

