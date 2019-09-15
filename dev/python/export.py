import json
from splinefit import iges
import pyiges.IGESGeomLib as IGES
from pyiges.IGESCore import IGEStorage
from pyiges.IGESGeomLib import IGESPoint

filename = "surface.igs"
system = IGEStorage()
iges.standard_iges_setup(system, filename)
path = 'output_regular_meshes/Garnet_Hill_Group1_3/'
path = 'PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5'
path1 = 'output_groups/Garnet_Hill_Group1_3/'
path2 = 'output_groups/Garnet_Hill_Group2_3/'
files = ['part_0_bspline_surf_fit.json',
         'part_0_bspline_surf_fit_bottom.json', 
         'part_0_bspline_surf_fit_left.json', 
         'part_0_bspline_surf_fit_right.json', 
         'part_0_bspline_surf_fit_top.json']

surf=[]

def surface(path, filename):
    with open(path + filename) as json_file:
        data = json.load(json_file)
        #surf.append()
        Px = data['real_world_Px']
        Py = data['real_world_Py']
        Pz = data['real_world_Pz']
        U = data['U']
        V = data['V']
        pu = data['pu']
        pv = data['pv']
        iges_surface = iges.IGESBSplineSurface(Px, Py, Pz, U, V, pu, pv)

    system.Commit(iges_surface)


def curve(idx):
    with open(path + files[idx]) as json_file:
        data = json.load(json_file)
        Px = data['real_world_Px']
        Py = data['real_world_Py']
        Pz = data['real_world_Pz']
        U = data['U']
        pu = data['p']
        iges_curve = iges.IGESBSplineCurve(Px, Py, Pz, U, pu)

    system.Commit(iges_curve)

surface(path1, files[0])
#surface(path2, files[0])
#for idx in range(1, 5):
#    curve(idx)
print("Saving")
system.save(filename)


