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
