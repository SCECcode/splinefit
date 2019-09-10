from mba import *
import matplotlib.pyplot as plt
import mba
import numpy as np
import pickle
import splinefit as sf
import helper

pu = 3
pv = 3
nu = 30
nv = 30
mu = 4
mv = 4
u  = np.random.uniform(0, 1, (nu,nv)).flatten()
v  = np.random.uniform(0, 1, (nu,nv)).flatten()
z = np.cos(10*v)*np.sin(10*u)*np.exp(-10*((u-0.5)**2 + (v-0.5)**2))
z = np.exp(-10*((u-0.5)**2 + (v-0.5)**2))

U = sf.bspline.uniformknots(mu, pu)
V = sf.bspline.uniformknots(mv, pv)

Px = np.linspace(0, 1, mu)
Py = np.linspace(0, 1, mv)

S = sf.bspline.Surface(U, V, pu, pv, Px, Py, 0*Px)


Pz, res = sf.bspline.lsq2surf(u, v, z, U, V, pu, pv)

S.eval(2*nu,2*nv)

def plot_surface(S):
    ax = helper.plot_grid(S.X, S.Y, S.Z)
    ax = helper.plot_grid(S.Px, S.Py, 0*S.Pz, ax, color='C0')



#plt.plot(u, v)
plt.scatter(x=u, y=v, c=z, cmap='RdBu')

#plt.scatter(x=u, y=v, c=Pz, cmap='RdBu')
plot_surface(S)
plt.show()
