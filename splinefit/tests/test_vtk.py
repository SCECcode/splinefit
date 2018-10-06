from splinefit import msh, vtk
import numpy as np

def load():
    n, e = msh.read('fixtures/test.msh')
    e = msh.get_data(e, num_members=3)
    return n, e


def test_write_triangular_mesh():
    p, t = load()
    vtk.write_triangular_mesh('mesh/test.vtk', p[:,1:], t)

def test_write_surface():
    p, t = load()
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(0.4*np.pi*X)*np.cos(0.5*np.pi*Y)
    vtk.write_surface('mesh/surf.vtk', X, Y, Z)
