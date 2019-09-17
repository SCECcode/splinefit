"""
usage:
    fit_boundary input=string output=string --verbose=int --degree=int \
                 --regularization=int --num_knots=int --estimate-knots=int \
                 --showfig=int --savefig=string

    fit_boundary --help

    Perform BSpline fitting to boundary segments.

    Required arguments:

     inputfile:     Load pickle binary data file containing boundary data
    outputfile:     Write pickle binary data file

    Optional arguments:

    help:           Display this information.    
    verbose:        Give detailed output (default: 1)
    degree:         Degree of BSpline basis functions
    regularization: Strength of regularization term
    num_knots:      Number of knots to try
    estimate-knots: Automatically determine the number of knots to use 
                    (default: 1)
    showfig:        Show a figure for each fitted BSpline curve
    savefig:        Write figure to png-file. Specify path, 
                    excluding file extension
"""
import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt
import bspline_boundary_functions as bfun

def main():
    options = get_options(sys.argv)

    data = pickle.load(open(options.inputfile, 'rb'))

    bspline_curves = []

    for num, bnd in enumerate(data.boundaries):
        min_degree = sf.bspline.min_degree(len(bnd.x), options.p)
        if options.estimate_knots:
            num_knots = estimate_knots(bnd.x, bnd.y, bnd.z)
        else:
            num_knots = options.num_knots
        vprint("Processing curve %d: degree = %d knots = %d " % 
               (num + 1, min_degree, num_knots), options.verbose)
        curve, res = fit_curve(bnd.x, bnd.y, bnd.z, min_degree, num_knots,
                               a=options.regularization)
        vprint("    Residual: %g " % res, options.verbose)
        bspline_curves.append(curve)

        if options.savefig:
            savefig= "%s%d.png" % (options.savefig, num)
        else:
            savefig = ""

        make_plot(bspline_curves[-1], savefig=savefig,
                  showfig=options.showfig)

    
    data.bspline_curves = bspline_curves
    pickle.dump(data, open(options.outputfile, 'wb'))


def vprint(msg, verbose):
    if not verbose:
        return
    print(msg)


def get_options(argv):
    """
    Get command line arguments.
    """

    options = helper.Struct()
    if '--help' in argv:
        print(helptxt)
        exit()

    args = helper.get_args(argv)
    options.inputfile = args['input']
    options.outputfile = args['output']

    if '--degree' in args:
        options.p = int(args['--degree'])
    else:
        options.p = 3

    if '--verbose' in args:
        options.verbose = int(args['--verbose'])
    else:
        options.verbose = 0


    if '--regularization' in args:
        options.regularization = float(args['--regularization'])
    else:
        options.regularization = 0.2
    
    if '--showfig' in args:
        options.showfig = int(args['--showfig'])
    else:
        options.showfig = 0

    if '--savefig' in args:
        options.savefig = args['--savefig']
    else:
        options.savefig = ''

    if '--num_knots' in args:
        options.num_knots = int(args['--num_knots'])
    else:
        options.num_knots = 10

    if '--estimate_knots' in args:
        options.estimate_knots = int(args['--estimate_knots'])
    else:
        options.estimate_knots = 1

    return options


def estimate_knots(x, y, z):
    """
    Estimate the number of knots by calculating the length of the piecewise
    linear segment and the average spacing between segments.
    """

    t = sf.bspline.chords(x, y, z)

    diff = t[1:] - t[0:-1]
    return int(1 / np.mean(diff))


def fit_curve(x, y, z, p, m, a=0.5, tol=1e-6):
    """
    Fit BSpline curve using linear least square approximation with second
    derivative regularization.
    """

    xm = np.mean(x)
    ym = np.mean(y)
    zm = np.mean(z)
    t = sf.bspline.chords(x-xm, y-ym)
    U = sf.bspline.uniformknots(m, p)
    l = np.zeros((len(U) - p - 1,))
    wx = 1.0 + 0 * l
    wy = 1.0 + 0 * l
    wz = 1.0 + 0 * l
    Px, rx = sf.bspline.lsq(t, x - xm, U, p, tol=tol, s=0, a=a, w=wx)
    Py, ry = sf.bspline.lsq(t, y - ym, U, p, tol=tol, s=0, a=a, w=wy)
    Pz, rz = sf.bspline.lsq(t, z - zm, U, p, tol=tol, s=0, a=a, w=wz)

    curve = helper.Struct()
    curve.x = x
    curve.y = y
    curve.z = z
    curve.Px = Px + xm
    curve.Py = Py + ym
    curve.Pz = Pz + zm
    curve.U = U
    curve.p = p
    curve.u = t
    curve.px = curve.Px[:-p]
    curve.py = curve.Py[:-p]
    curve.int_knot = m

    res = rx + ry + rz
    return curve, res


def make_plot(curve, savefig="", showfig=0, npts=100, color=1):
    if not savefig and not showfig:
        return
    cx, cy, cz = helper.evalcurve3(curve, npts)

    plt.clf()
    plt.plot(curve.x,curve.y,'k-o')
    plt.plot(cx,cy,'C%d-'%color)
    plt.plot(curve.Px, curve.Py, 'C%do-'%color, alpha=0.3)

    if savefig:
        plt.savefig(savefig)

    if showfig:
        plt.show()

if __name__ == "__main__":
    main()
