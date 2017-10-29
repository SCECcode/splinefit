from splinefit import gmsh

def load(test):

    txt = open('fixtures/' + test + '.geo').read() 
    return txt

def test_get_command():

    geo = "Point(1) = {1.0, 2.0, 3.0};"
    point = gmsh.get_command('Point', geo)
    assert point['1'][0] == '1.0'

    tests = ['test1']

    for test in tests:
        splines = gmsh.get_command('Spline', load(test))
        points  = gmsh.get_command('Point', load(test))
        assert gmsh.check_groupmembers(splines, points)

def test_read():
    from six import iteritems

    tests = ['fixtures/test1.geo']

    for test in tests:
        var, cmd = gmsh.read(test)

def test_check_group():

    a = {0 : [0], 1 : [1]}
    b = {0 : 1, 1 : 1}
    assert gmsh.check_groupmembers(a, b)
    b = [0, 1]
    assert gmsh.check_groupmembers(a, b)
    a = {0 : [0], 1 : [1, 2]}
    assert not gmsh.check_groupmembers(a, b)

def test_get_variables():

    geo = 'a = 10;'
    var = gmsh.get_variables(geo)
    assert var['a'] == '10'
    geo = 'p2 = 10.0 + p2;'
    var = gmsh.get_variables(geo)
    assert var['p2'] == '10.0 + p2'
    geo = '2p = 10.0 + p2;'
    var = gmsh.get_variables(geo)
    assert var == None
    geo = 'p_2 = 1 - 2;'
    var = gmsh.get_variables(geo)
    assert var['p_2'] == '1 - 2'
    geo = 'p_2 = 1 - 2;\na = b;'
    var = gmsh.get_variables(geo)
    assert var['p_2'] == '1 - 2'
    assert var['a'] == 'b'

def test_counter():

    c = gmsh.Counter()
    assert c['newp'] == 0
    assert c['newp'] == 1
    assert c['newl'] == 0

def test_eval_vars():

    gmsh.counter.reset()
    variables = {'a' : 'newp', 'b' : 'newl', 'c': 'newp'}
    newvars = gmsh.eval_vars(variables)
    assert newvars['a'] == 0
    assert newvars['b'] == 0
    assert newvars['c'] == 1

def test_subs():

    gmsh.counter.reset()
    variables = {'a' : 'newp', 'b' : 'newl', 'c': 'newp'}
    newvars = gmsh.eval_vars(variables)
    groups = {'g0' : ['b', 'a'], 'g1' : ['b', 'c']}
    newgroups = gmsh.subs(groups, newvars)

    assert newgroups['g0'][0] == '0' 
    assert newgroups['g0'][1] == '0' 
    assert newgroups['g1'][0] == '0' 
    assert newgroups['g1'][1] == '1' 

    groups = {'g0 + a' : ['b', 'a'], 'g1 + b' : ['b', 'c']}
    newgroups = gmsh.subs(groups, newvars)
    assert newgroups['g0 + a'] 

def test_eval_groups():

    gmsh.counter.reset()
    variables = {'a' : 'newp', 'b' : 'newl', 'c': 'newp'}
    newvars = gmsh.eval_vars(variables)
    groups = {'g0' : ['b', 'a'], 'g1' : ['b', 'c']}
    groups = gmsh.subs(groups, newvars)
    groups = gmsh.eval_groups(groups, grouptype='Spline')

    assert groups['g0'][0] == '0' 
    assert groups['g0'][1] == '0' 
    assert groups['g1'][0] == '0' 
    assert groups['g1'][1] == '1' 

def test_write():

    var, cmd = gmsh.parse('fixtures/test1.geo')
    gmsh.write('fixtures/out.geo', var, cmd)

def test_write_command():

    groups = {'0' : [0.0, 1.0]}
    cmdstr = gmsh.write_command('Spline', groups)
    assert cmdstr == 'Spline(0) = {0.0, 1.0};\n'
    groups = [[0.0, 1.0]]
    cmdstr = gmsh.write_command('Spline', groups)
    assert cmdstr == 'Spline(0) = {0.0, 1.0};\n'

def test_write_variables():

    var = {'a' : 1.0 }
    assert gmsh.write_variables(var) == 'a = 1.0;\n'

def test_parse():

    var, grp = gmsh.parse('fixtures/test1.geo')
    print var
    for g in grp:
        print g, grp[g]
    assert False

def test_get_group():

    geo = 'Ruled Surface(1) = {1, 2, 3};'
    grp = gmsh.get_group(geo)
    assert grp['type'] == 'Ruled Surface'
