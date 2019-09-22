import pyiges.IGESGeomLib as IGES
from pyiges.IGESCore import IGEStorage
from pyiges.IGESGeomLib import IGESPoint
from splinefit import iges, msh
import json

filename = "garnet_hill"
filename = "compton"
filename = "newport"
system = IGEStorage()
iges.standard_iges_setup(system, filename + ".igs")
path = lambda num: 'output_regular_meshes_cut/Garnet_Hill_Group%d_3/' % num
path = lambda num: 'output_regular_meshes_cut/PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2/'
path = 'output_regular_meshes_cut/PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5/'
surface_file = lambda num: 'part_%d_bspline_surf_fit.json' % num


newpath='output3/PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5/'
surface_file2 = lambda num: '%s/pydata/surf_%d.p' % (newpath, num)
boundary_files = lambda num : \
                     ['part_%d_bspline_surf_fit_bottom.json' % num, 
                      'part_%d_bspline_surf_fit_left.json' % num, 
                      'part_%d_bspline_surf_fit_right.json' % num, 
                      'part_%d_bspline_surf_fit_top.json' % num]

def load(num, path):
    import pickle

    data = pickle.load(open(surface_file2(num+1), 'rb'))
    boundary = boundary_files(num)
    curves = [curve.iges() for curve in data['bspline_curves']]
    #surface = iges.load_surface(path, surface_file(num), system=system, real_world=True)
    for curve in curves:
        system.Commit(curve)
    surface = data.bspline_surface.iges()
    system.Commit(surface)
    #curves = iges.load_curves(path, boundary_files(num), system=system, real_world=True)
    boundary, bounded_surface = iges.build_bounded_surface(surface, curves,
                                                           system=system)

num_surfaces = 7

for num in range(num_surfaces):
    load(num, path)
system.save(filename + ".igs")
msh.write_geo(filename, num_surfaces)

