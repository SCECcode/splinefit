# This function used be in bspline-surface
def tmp():
    #plt.clf()
    #plt.plot(xt, zt)

    #S.Pz[0,:] = 3
    #S.Pz[-1,:] = 3
    #S.Pz[:,0] = 3
    #S.Pz[:,-1] = 3
    from scipy.optimize import least_squares
    from scipy.optimize import minimize
    m = len(S.Pz[-1,:])

    def f_min(x):
        Px = x[0:m]
        Py = x[m::]
        dists = sf.bspline.dist2curve(S.pu, S.U, Px, Py, ut, xt, zt)
        err = sum(dists)
        print(err)
        return err

    Px = S.Px[-1,:]
    curve = helper.Struct()
    curve.Py = 0
    def f_min2(x):
        curve.Py, res = sf.bspline.lsq(s, zt, S.U, S.pu, s=x[0])
        dists = sf.bspline.dist2curve(S.pu, S.U, Px, curve.Py, ut, xt, zt)
        err = sum(dists)
        print("Error:", err)
        return err

    x0 = np.zeros((2*m,))
    s = sf.bspline.chords(xt, zt, a=0, b=1)
    S.Pz[-1,:], res = sf.bspline.lsq(s, zt, S.U, S.pu)
    print(res)
    #S.Px[-1,:], S.Pz[-1,:], U, res = sf.bspline.lsq2l2(xt, zt,
    #        len(S.U)-2*S.pu-2, S.pu, knots='uniform', smooth=0.0)
    S.Px[-1,:], res = sf.bspline.lsq(s, xt, S.U, S.pu)
    x0[0:m] = S.Px[-1,:]
    x0[m::] = S.Pz[-1,:]
    print(x0.shape)

    x0 = 10
    res = minimize(f_min2, [x0], method='Nelder-Mead', options={'maxiter': 300,
        'disp':True}, tol=1e-2)
    #res = minimize(f_min, x0, options={'maxiter': 300, 'disp':True})
    #res = least_squares(f_min, x0)
    print(res)

    #curve.Px = S.Px[-1,:]
    #curve.Py = S.Pz[-1,:]
    #curve.Px = res.x[0:m]
    #curve.Py = res.x[m::]
    curve.Px = Px

    #S.Pz[-1,:], res = sf.bspline.lsq(ut, zt, S.U, S.pu)
    curve.p = S.pu
    curve.U = S.U
    cx, cy = helper.evalcurve(curve, 100)
    dists = sf.bspline.dist2curve(S.pu, S.U, curve.Px, curve.Py, ut, xt, zt)
    print(dists.shape)
    print(sum(dists))
