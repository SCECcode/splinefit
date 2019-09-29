from math import pi
from pyiges.IGESCore import IGESItemData, IGEStorage

class IGESWriter(IGEStorage):
    def save(self, filename = 'IGESFile.igs'):
        try:
            myFile = open(filename, 'w')
            #myFile.write(str(self))
            myFile.write(str(self.StartSection))
            myFile.write(str(self.GlobalSection))
            myFile.write(str(self.DirectorySection))
            myFile.write(str(self.ParameterSection))
            myFile.write(str("\n"))
            myFile.write(str(self.IGESTerminate()))
            myFile.close()
        except:
            print("Failed to write: %s" % filename)


def standard_iges_setup(system, filename):
    system.StartSection.Prolog = " "
    
    system.GlobalSection.IntegerBits = int(32)
    system.GlobalSection.SPMagnitude = int(38)
    system.GlobalSection.SPSignificance = int(6)
    system.GlobalSection.DPMagnitude = int(38)
    system.GlobalSection.DPSignificance = int(15)
    
    system.GlobalSection.MaxNumberLineWeightGrads = int(8)
    system.GlobalSection.WidthMaxLineWeightUnits = float(0.016)
    system.GlobalSection.MaxCoordValue = float(71)

    index_dot = filename.index('.')
    system.GlobalSection.ProductIdentificationFromSender = filename[:index_dot]
    system.GlobalSection.FileName = filename
    
    system.GlobalSection.ProductIdentificationForReceiver = \
      system.GlobalSection.ProductIdentificationFromSender
      
    system.GlobalSection.AuthorOrg = "USC"
    system.GlobalSection.NameOfAuthor = "Ossian O'Reilly"

class IGESBSplineSurface(IGESItemData):
    def __init__(self, Px, Py, Pz, U, V, pu, pv):
        IGESItemData.__init__(self)
        self.LineFontPattern.setSolid()
        self.LineWeightNum = 1
        self.FormNumber = 0

        self.EntityType.setRBSplineSurface() # 128

        K1 = len(Px[0]) - 1
        K2 = len(Px) - 1
        M1 = pu # Degree of first set of basis functions
        M2 = pv # Degree of second set of basis functions
        prop1 = 0 # PROP1 Not closed
        prop2 = 0 # PROP2 Not closed
        prop3 = 1 # PROP3 Polynomial
        prop4 = 0 # PROP4 Non-periodic in first parametric variable direction
        prop5 = 0 # PROP5 Non-periodic in second parametric variable direction

        Px = [item for sublist in Px for item in sublist]
        Py = [item for sublist in Py for item in sublist]
        Pz = [item for sublist in Pz for item in sublist]
        W  = [1 for Pi in Px]

        nodes = []
        for Pi, Pj, Pk in zip(Px, Py, Pz):
            nodes.append([Pi, Pj, Pk])

        nodes = [item for sublist in nodes for item in sublist]

        ls = [K1, K2, M1, M2, prop1, prop2, prop3, prop4, prop5] \
              + U +  V + W + nodes + [0, 1, 0, 1]
        self.AddParameters(ls)


class IGESBSplineCurve(IGESItemData):
    def __init__(self, Px, Py, Pz, U, pu):
        IGESItemData.__init__(self)
        self.LineFontPattern.setSolid()
        self.LineWeightNum = 1
        self.FormNumber = 0
        self.sense = 0

        self.EntityType.setRBSplineCurve() # 126
 
        K = len(Px) - 1
        M = pu # Degree of basis functions
        prop1 = 0 # PROP1 Not closed
        prop2 = 0 # PROP2 Not closed
        prop3 = 1 # PROP3 Polynomial
        prop4 = 0 # PROP4 Non-periodic

        W  = [1.0 for Pi in Px]

        N = 1+K-M

        nodes = []
        for Pi, Pj, Pk in zip(Px, Py, Pz):
            nodes.append([Pi, Pj, Pk])

        nodes = [item for sublist in nodes for item in sublist]

        ls = [K, M, prop1, prop2, prop3, prop4] + U + W + nodes + [0, 1]
        self.nodes_start = 6 + len(U) + len(W)
        self.AddParameters(ls)

class IGESBoundary(IGESItemData):
    def __init__(self, surface, boundary_curves):
        IGESItemData.__init__(self)
        self.EntityType.setBoundary() # 141

        boundary_type = 1 
        preference = 2
        self.N1 = len(boundary_curves)
        ls = [boundary_type, preference, 
              surface.DirectoryDataPointer.data,
              self.N1]
        for curve in boundary_curves:
            ls += [curve.DirectoryDataPointer.data, curve.sense, 1,
                    curve.ParameterDataPointer.data]
        self.AddParameters(ls)


class IGESBoundedSurface(IGESItemData):
    def __init__(self, bounded_surface, boundaries):
        IGESItemData.__init__(self)
        self.EntityType.setBoundedSurface()

        self.N1 = len(boundaries)

        try:
            outer_boundary_surface = outer_boundary_surface.DirectoryDataPointer.data
        except:
            pass

        self.AddParameters([1, 
                            bounded_surface.DirectoryDataPointer.data,
                            self.N1]
                           )

        for boundary in boundaries:
            self.AddParameters([boundary.DirectoryDataPointer.data])

def load_curves(path, files, real_world=False, system=None):
    import json
    curves = []
    for filei in files:
        with open(path + filei) as json_file:
             data = json.load(json_file)
             if real_world:
                 Px = data['real_world_Px']
                 Py = data['real_world_Py']
                 Pz = data['real_world_Pz']
             else:
                 Px = data['Px']
                 Py = data['Py']
                 Pz = data['Pz']
             U = data['U']
             pu = data['p']
             curve = IGESBSplineCurve(Px, Py, Pz, U, pu)
             curves.append(curve)
             if system:
                system.Commit(curve)
    return curves

def load_surface(path, filename, real_world=False, system=None):
    import json
    with open(path + filename) as json_file:
        data = json.load(json_file)
        if real_world:
            Px = data['real_world_Px']
            Py = data['real_world_Py']
            Pz = data['real_world_Pz']
        else:
            Px = data['Px']
            Py = data['Py']
            Pz = data['Pz']
        U = data['U']
        V = data['V']
        pu = data['pu']
        pv = data['pv']
        surface = IGESBSplineSurface(Px, Py, Pz, U, V, pu, pv)
        if system:
            system.Commit(surface)

    return surface

def build_bounded_surface(surface, curves, system=None):
    boundary = IGESBoundary(surface, curves)
    if system:
        system.Commit(boundary)
    bounded_surface = IGESBoundedSurface(surface, [boundary])
    if system:
        system.Commit(bounded_surface)
    return boundary, bounded_surface
