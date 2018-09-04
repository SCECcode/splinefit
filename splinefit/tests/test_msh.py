import numpy as np
from splinefit import msh

def load(test='test'):
    txt = open('fixtures/' + test + '.msh').read() 
    return txt

def test_read():
    n, e = msh.read('fixtures/test.msh')

def test_write():
    n, e = msh.read('fixtures/test.msh')
    msh.write('fixtures/new.msh', n, e)

def test_nodes():
    txt = load()

    nodes = msh.nodes(txt)

    assert nodes[0][0] == 1
    assert nodes[0][1] == 9.1881354387402361
    assert nodes[0][2] == 4.551920309100666
    assert nodes[0][3] == 2.866023898322641

def test_elements():
    txt = load()

    elems = msh.elements(txt)

    # 1 15 2 0 5 1
    assert elems[0][0] == 1
    assert elems[0][1] == 15
    assert elems[0][2] == 2
    assert elems[0][3] == 0
    assert elems[0][4] == 5
    assert elems[0][5] == 1

def test_get_data():
    txt = load()
    elems = msh.elements(txt)
    tris = msh.get_data(elems, num_members=3, index=1)
    assert tris.shape[1] == 3
    assert tris[0][0] == 0
    assert tris[0][1] == 5
    assert tris[0][2] == 1




