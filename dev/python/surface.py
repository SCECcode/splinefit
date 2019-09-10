from mba import *
from numpy import *
from matplotlib.pyplot import *
import mba
import numpy as np
import pickle


S, data =\
pickle.load(open('../output_group/Northridge_3/pydata/part_0_bspline_surf.p',
    'rb'))
 
cmin = [0.0, 0.0]
cmax = [1.0, 1.0]
coo  = np.random.uniform(0, 1, (7,2))
val  = np.random.uniform(0, 1, coo.shape[0])

n = 100
s = linspace(0,1,n)
x = array(meshgrid(s,s)).transpose([1,2,0]).copy()


def initial_surface():

    pcolormesh(data.u, data.v, data.z, cmap='RdBu')

initial_surface()
show()
exit(1)

 
def plot_surface(u0, v0):
    interp = mba2(cmin, cmax, [u0,v0], coo, val)
    error = amax(abs(val - interp(coo))) / amax(abs(val))
    v = interp(x)
    pcolormesh(s, s, v, cmap='RdBu')
    scatter(x=coo[:,0], y=coo[:,1], c=val, cmap='RdBu')
    xlim([0,1])
    ylim([0,1])
    title("$m_0 = {0:} {1:}$, error = {2:.3e}".format(u0, v0, error))
    colorbar();

figure(figsize=(11,5))
u = 4
v = 6
u2 = 6
v2 = 8
subplot(121); plot_surface(u, v)
subplot(122); plot_surface(u2, v2)
tight_layout()
show()
