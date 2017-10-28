from splinefit import gmsh

def load(test):
    txt = open('fixtures/' + test + '.geo').read() 
    return txt

def test_get_command():

    tests = ['test1']

    for test in tests:
        splines = gmsh.get_command('Spline', load(test))
        points  = gmsh.get_command('Point', load(test))
        gmsh.check_groupmembers(splines, points)

    



