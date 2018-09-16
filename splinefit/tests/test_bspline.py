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

    n = 5
    t = np.linspace(0, 1, n)
    x = np.zeros(((len(P)-3)*n,))
    y = np.zeros(((len(P)-3)*n,))
    k = 0
    for i in range(1,len(P)-2):
        ctrlpts = P[i-1:i+3]
        for ti in t:
            qx = sf.bspline.cubic(ti).dot(ctrlpts)
            y[k] = qx
            x[k] = i + ti
            k += 1
    plt.clf()
    plt.plot(x,y)
    #plt.show()





