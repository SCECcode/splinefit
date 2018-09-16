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
    pass
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
        z.append(sf.bspline.curvepoint(nctrl, p, U, P, ui))
    spl = make_lsq_spline(px, py, U, p)
    plt.clf()
    plt.plot(px, py, 'bo')
    plt.plot(u, spl(u), 'g-', lw=3, label='LSQ spline')
    plt.plot(u, z,'k')
    plt.show()

def test_bspline_curve():

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

    Px = sf.bspline.bspline_lsq(px, py, U, p)
    Py = sf.bspline.bspline_lsq(py, px, U, p)

    zx = []
    zy = []
    u = np.linspace(-3,3,100)
    for ui in u:
        zx.append(sf.bspline.curvepoint(nctrl, p, U, Px, ui))
        zy.append(sf.bspline.curvepoint(nctrl, p, U, Py, ui))

    plt.plot(zx, zy,'ko')
    plt.show()
