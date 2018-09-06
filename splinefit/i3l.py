"""
Implementation of the 3L algorithm

The 3L Algorithm for Fitting Implicit
Polynomial Curves and Surfaces to Data
Michael M. Blane, Zhibin Lei, Member, IEEE, Hakan CË ivi, and David B. Cooper, Fellow, IEEE
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 22, NO. 3, MARCH 2000

"""
import numpy as np

def vander2d(p, deg):
    """
    Construct Vandermonde-like matrix

    m = [1 x y]
    m = [1 x x**2 y x*y y**2]
       = [(0,0) (1,0) (2,0) (1,1) (0,1), (0,2)]


         (0,0) (1,0) (2,0)
         (0,1) (1,1) 
         (0,2)

       = [(0,0) (1,0) (2,0) (3,0)
          (0,1) (1,1) (2,1) 
          (0,2) (1,2) 
          (0,3)
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
    coef = np.linalg.lstsq(M, b, rcond=None)
    return coef
