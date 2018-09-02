# Splinefit

Experimental python package for building a geometric description of a small subset of the [Community Fault Model](https://scec.usc.edu/scecpedia/CFM) using splines. 

# Installation
```bash
$ git clone https://github.com/ooreilly/splinefit
$ python [sudo] setup.py install [--user]
```

# Commandline tools

## Convert GOCAD tsurf to gmsh
Use `tsurfmsh` to convert a triangular mesh stored in the `.tsurf` file format to the gmsh `.msh` file format. 
```bash
$ tsurfmsh (input) (output)

```

# Parsers

## Gmsh .geo
The module `gmsh` provides a very limited parser for gmsh geo files. This parser is currently restricted to only reading
points, lines, splines, and surfaces. More advanced functionality such as loops, and many of the gmsh commands are
not supported.

**example.geo**
```python
cl__1 = 1e+22;
Point(1) = {0.5, -0.5, 0, cl__1};
Point(2) = {-0.5, -0.5, 0, cl__1};
Point(3) = {-0.5, 0.5, 0, cl__1};
Point(4) = {0.5, 0.5, 0, cl__1};
Point(5) = {0, 0.7, 0, cl__1};
Point(6) = {0.7, -0, 0, cl__1};
Point(7) = {-0, -0.7, 0, cl__1};
Point(8) = {-0.7, -0, 0, cl__1};
Spline(1) = {3, 5, 4};
Spline(2) = {4, 6, 1};
Spline(3) = {1, 7, 2};
Spline(4) = {2, 8, 3};
Line Loop(1) = {1, 2, 3, 4};
Surface(1) = {1};
```

**example.py**
```python
In [1]: from splinefit import gmsh

In [2]: var, obj = gmsh.read('example.geo')

In [3]: var
Out[3]: {'cl__1': '1e+22'}

In [4]: obj
Out[4]: 
{'Line Loop': {1: [1, 2, 3, 4]},
 'Point': {1: [0.5, -0.5, 0.0, 1e+22],
  2: [-0.5, -0.5, 0.0, 1e+22],
  3: [-0.5, 0.5, 0.0, 1e+22],
  4: [0.5, 0.5, 0.0, 1e+22],
  5: [0.0, 0.7, 0.0, 1e+22],
  6: [0.7, 0.0, 0.0, 1e+22],
  7: [0.0, -0.7, 0.0, 1e+22],
  8: [-0.7, 0.0, 0.0, 1e+22]},
 'Spline': {1: [3, 5, 4], 2: [4, 6, 1], 3: [1, 7, 2], 4: [2, 8, 3]},
 'Surface': {1: [1]}}

In [5]: gmsh.write('out.geo', var, obj)

```
**out.geo**
```python
SetFactory("OpenCASCADE");
cl__1 = 1e+22;
Point(1) = {0.5, -0.5, 0.0, 1e+22};
Point(2) = {-0.5, -0.5, 0.0, 1e+22};
Point(3) = {-0.5, 0.5, 0.0, 1e+22};
Point(4) = {0.5, 0.5, 0.0, 1e+22};
Point(5) = {0.0, 0.7, 0.0, 1e+22};
Point(6) = {0.7, 0.0, 0.0, 1e+22};
Point(7) = {0.0, -0.7, 0.0, 1e+22};
Point(8) = {-0.7, 0.0, 0.0, 1e+22};
Spline(1) = {3, 5, 4};
Spline(2) = {4, 6, 1};
Spline(3) = {1, 7, 2};
Spline(4) = {2, 8, 3};
Line Loop(1) = {1, 2, 3, 4};
Surface(1) = {1};
```
Note that there are some differences between `example.geo` and `out.geo`. In particular, `out.geo` starts with
`SetFactory("OpenCASCADE")` this command tells gmsh to use the OpenCascade geometry and that is required to be able to
export the geometry to the OpenCASCADE file formats `.brep`. and `.step`. Besides that, in `out.geo` all the variables 
have been evaluated.

# Tests
The directory `splinefit/tests` contains a series of tests that can be run using [pytest](https://docs.pytest.org/en/latest/)
```bash
$ pytest splinefit/tests
```
To ensure that some test data can be found it is important that the tests are run from the root directory as indicated above.
The tests are also useful to study to see examples of how to use different the modules.

