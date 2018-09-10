"""
Implementation of the 3L algorithm

The 3L Algorithm for Fitting Implicit
Polynomial Curves and Surfaces to Data
Michael M. Blane, Zhibin Lei, Member, IEEE, Hakan CE ivi, and David B. Cooper,
Fellow, IEEE
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 22, NO. 3,
MARCH 2000

"""
import numpy as np

def vander2d(p, deg):
    """
    Construct Vandermonde-like matrix that arises from treating the coefficients 
    a_ij as unknowns when representing the bivariate polynomial graph

    f(x,y) = sum_ij a_ij x^i y^j,  i + j < deg + 1
           = m*a

    For deg = 2, coefficients are ordered as:
    m = [1 x x**2 y x*y y**2]
         (0,0) (1,0) (2,0)
         (0,1) (1,1) 
         (0,2) 
         ] 
    and for deg = 3, 
       = [(0,0) (1,0) (2,0) (3,0)
          (0,1) (1,1) (2,1) 
          (0,2) (1,2) 
          (0,3)]
    and so forth.

    Arguments:
        p : Array of points (size: num pts x 2)
        deg : Degree of bivariate polynomial

    Returns:
        M : Vandermonde-like matrix. M[0,0] = 1, M[0,1] = x_i (the x-coordinate
            of the first point).

    """
    npts = p.shape[0]

    ncols = int( (deg+1)**2 - 0.5*((deg+1)**2 - (deg+1)))
    M = np.zeros((npts, ncols))

    for i in range(npts):
        l = 0
        for j in range(deg+1):
            for k in range(deg+1):
                if j + k < deg + 1:
                    M[i,l] = p[i,0]**k*p[i,1]**j
                    l = l + 1

    return M

def solve(M0, Mp, Mm, c):
    """

    Solve the least squares problem:

     min || M*x - b||^2

      M =  [ Mp     b = [ c
             M0           0
             Mm ]        -c ]

      M0 is the zero level set curve that approximates the point cloud, 
      Mp is the +c level set curve that approximates the point cloud displaced
      in the positive normal direction by a small amount.
      Mm is the -c level set curve that approximates the point cloud displaced
      in the negative normal direction by a small amount.

    """
    M = np.vstack((Mp, M0, Mm))
    e = np.ones((M0.shape[0],1))
    b = np.vstack((c*e, 0*e, -c*e))
    coef = np.linalg.lstsq(M, b, rcond=False)
    return coef

# f(x,y) = c
# a1 + a2*x + a3*x^2 + a4*y + a5*x*y + a6*y**2
