import splinefit as sf
import matplotlib.pyplot as plt
import numpy as np

def test_basis():
    n = 20
    s = np.linspace(0,1,n)

    x = np.zeros((n,4))
    y = np.zeros((n,4))

    for i, si in enumerate(s):
        z = sf.bspline.basis(si)
        for j in range(4):
            x[i,j] = si+2 - j
            y[i,j] = z[j]

    plt.plot(x,y)
    #plt.show()




