from splinefit import msh, vtk
import numpy as np

def load():
    n, e = msh.read('fixtures/test.msh')
    e = msh.get_data(e, num_members=3)
    return n, e


def test_write_triangular_mesh():
    p, t = load()
    vtk.write_triangular_mesh('fixtures/test.vtk', p[:,1:], t)
