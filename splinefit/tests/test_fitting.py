import numpy as np
import splinefit as sf

def test_pca():
    # Check the PCA implementation by passing points that lie in the plane z= 0 and
    # that output is equal to the standard basis in R^2.
    points = np.array([[0.0, 0.0, 0.0],
                       [1.0,0.0,0.0],
                       [1.0,1.0,0.0],
                       [0.0,1.0,0.0]])

    num_dims = 3
    num_comp = 2
    eig_vec = sf.fitting.pca(points, num_components=num_comp)
    assert eig_vec.shape[0] == num_dims
    assert eig_vec.shape[1] == num_comp
    assert np.all(np.equal(eig_vec[:,0], np.array([1,0,0])))
    assert np.all(np.equal(eig_vec[:,1], np.array([0,1,0])))

def test_projection():
    points = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    basis = np.array([[1.0, 0.0],[0.0, 1.0], [0.0, 0.0]])

    proj = sf.fitting.projection(points, basis)

def test_normalize():
    points = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    xn, mu, std = sf.fitting.normalize(points)

    points_restored = sf.fitting.renormalize(xn, mu, std)
    assert np.all(np.isclose(points,points_restored))

def test_bbox2():

    points = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    bbox = sf.fitting.bbox2(points)
    assert bbox[0,0] == 0.0
    assert bbox[0,1] == 0.0
    assert bbox[1,0] == 1.0
    assert bbox[1,1] == 0.0
    assert bbox[2,0] == 1.0
    assert bbox[2,1] == 1.0
    assert bbox[3,0] == 0.0
    assert bbox[3,1] == 1.0

def test_bbox2_area():

    points = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    bbox = sf.fitting.bbox2(points)
    assert sf.fitting.bbox2_vol(bbox) == 1.0

def test_sdtriangle2():
    p0 = np.array((0.0, 0.0))
    p1 = np.array((-0.5, 0.0))
    p2 = np.array((0.0, 0.5))

    p = np.array((0.25, 0.25))
    dist = sf.fitting.sdtriangle2(p, p0, p1, p2)
    assert dist > 0

def test_sdquad2():
    p0 = np.array((0.0, 0.0))
    p1 = np.array((1.0, 0.0))
    p2 = np.array((1.0, 1.0))
    p3 = np.array((0.0, 1.0))
    d = lambda x, y : sf.fitting.sdquad2(np.array((x,y)), p0, p1, p2, p3)
    assert d(0.25,0.25) < 0
    assert d(1.10,1.15) > 0
    assert d(-0.10,0.0) > 0

def test_triangle2_vol():
    p0 = np.array((0.0, 0.0))
    p1 = np.array((1.0, 0.0))
    p2 = np.array((0.0, 1.0))
    area = sf.fitting.triangle_vol(p0, p1, p2)
    assert np.isclose(area, 0.5)

def test_quad_vol():
    p0 = np.array((0.0, 0.0))
    p1 = np.array((1.0, 0.0))
    p2 = np.array((1.0, 1.0))
    p3 = np.array((0.0, 1.0))
    area = sf.fitting.quad_vol(p0, p1, p2, p3)
    assert np.isclose(area, 1.0)

def test_argnearest():

    points = np.array([[0.0, 0.0, 0.0],
                       [1.0,0.0,0.0],
                       [1.0,1.0,0.0],
                       [0.0,1.0,0.0]])
    query = points[1,:] + 1e-2
    nearest = sf.fitting.argnearest(points, query)
    assert nearest == 1




