import sys
from splinefit import msh
import splinefit as sf
import numpy as np
import helper
import pickle
import matplotlib.pyplot as plt
import bspline_boundary_functions as bfun

inputfile = sys.argv[1]
outputfile = sys.argv[2]
p = int(sys.argv[3])
sm = float(sys.argv[4])
s = float(sys.argv[5])
refine_ratio = float(sys.argv[6])
regularization = float(sys.argv[7])
use_mapping = int(sys.argv[8])

if len(sys.argv) < 10:
    figfile = None
else:
    figfile = sys.argv[9]

if len(sys.argv) < 11:
    showplot = 0
else:
    showplot = int(sys.argv[10])
data = pickle.load(open(inputfile, 'rb'))

print("use mapping", use_mapping)
if use_mapping:
    print("Determining number of u-knots...")
    bottom, top = bfun.make_curves(data.bottom, data.top, p, sm, s, disp=True,
                                   axis=0, ratio=refine_ratio, a=regularization)
    print("Determining number of v-knots...")
    left, right = bfun.make_curves(data.left, data.right, p, sm, s, disp=True,
                                   axis=1, ratio=refine_ratio, a=regularization)
    mu = max(bfun.get_num_ctrlpts(left), bfun.get_num_ctrlpts(right))
    mv = max(bfun.get_num_ctrlpts(top), bfun.get_num_ctrlpts(bottom))
    
    print("Number of UV control points: [%d, %d]" % (mu, mv))
else:
    print("Determining number of knots for top boundary...")
    top = bfun.make_curve(data.top, p, sm, s, disp=True, axis=0, 
                          ratio=refine_ratio, a=regularization)
    print("Determining number of knots for bottom boundary...")
    bottom = bfun.make_curve(data.bottom, p, sm, s, disp=True, axis=0, 
                          ratio=refine_ratio, a=regularization)
    print("Determining number of knots for left boundary...")
    left = bfun.make_curve(data.left, p, sm, s, disp=True, axis=0, 
                          ratio=refine_ratio, a=regularization)
    print("Determining number of knots for right boundary...")
    right = bfun.make_curve(data.right, p, sm, s, disp=True, axis=0, 
                          ratio=refine_ratio, a=regularization)
    print("Number of control points")
    print("top:     %d " % bfun.get_num_ctrlpts(top))
    print("bottom:  %d " % bfun.get_num_ctrlpts(bottom))
    print("left:    %d " % bfun.get_num_ctrlpts(left))
    print("right:   %d " % bfun.get_num_ctrlpts(right))



bfun.make_plot(left, figfile, color=0)
bfun.make_plot(bottom, figfile, color=1)
bfun.make_plot(right, figfile, color=2)
bfun.make_plot(top, figfile, color=3, save=1)
helper.show(showplot)


pickle.dump((left, right, bottom, top, data), open(outputfile, 'wb'))
