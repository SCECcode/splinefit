from splinefit import tsurf

def load(test='Newport-Inglewood_fault'):
    txt = open('fixtures/' + test + '.ts').read() 
    return txt

def test_header():
    txt = load()

def test_version():
    txt = load()
    assert tsurf.version(txt) == '1'

def test_vrtx():
    txt = 'VRTX 9 399571.46875 3733843.25 -2499'
    vrtx = tsurf.vrtx(txt)
    assert vrtx[0][0]  == 9
    assert vrtx[0][1]  == 399571.46875
    assert vrtx[0][2]  == 3733843.25
    assert vrtx[0][3]  == -2499
    
    txt = 'VRTX 1 1.0 2.0 3.0\n'
    txt += 'VRTX 2 2.0 3.0 4.0\n'
    vrtx = tsurf.vrtx(txt)
    assert vrtx[0][0]  == 1
    assert vrtx[1][0]  == 2
    
def test_tri():
    txt = 'TRGL 49 46 47\n' 
    txt += 'TRGL 83 88 84\n' 
    txt += 'TRGL 23 22 20\n' 
    tri = tsurf.tri(txt)

    assert tri[0][0] == 49
    assert tri[1][0] == 83
    assert tri[2][0] == 23

def test_read():
    filename = 'fixtures/Newport-Inglewood_fault.ts'
    tsurf.read(filename)

def test_msh():
    from splinefit import msh
    filename = 'fixtures/Newport-Inglewood_fault.ts'
    p, t = tsurf.read(filename)

    e = tsurf.msh(t)
    msh.write('fixtures/Newport-Inglewood_fault.msh', p, e)



