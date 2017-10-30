supportedobjs = ['Point', 'Spline', 'Line Loop', 'Surface']
datatypes = {'Spline' : lambda x : int(x), 
             'Point'  : lambda x : float(x), 
             'Surface' : lambda x : int(x),
             'Line Loop' : lambda x : int(x)}
counters = {'Point' : 'newp', 'Line' : 'newl', 
            'Surface' : 'news', 'Volume' : 'newv', 
             'Line Loop' : 'newll'}

class Counter:

    def __init__(self):
        from six import iteritems
        self.counter = {}
        for k, v in iteritems(counters):
            self.counter[v] = 0

    def __getitem__(self, key):
        val = self.counter[key]
        self.counter[key] += 1
        return val

    def reset(self):
        for c in counters:
            self.counter[c] = 0

counter = Counter()

def get_object(geo):
    """
    Gets a gmsh object from geo script.

    Parameters

    geo : string,
          The geo script to parse

    Returns

    out : dict,
          contains the object `id`, `type`, and `data` if found. 
          When no object is found this function returns `none`.

    # Examples

    >>> point = get_object('Point(1) = {1.0, 2.0, 3.0};')
    >>> point['id']
    '1'
    >>> point['type']
    'Point'
    >>> point['data']
    ['1.0', '2.0', '3.0']

    """

    import re

    pattern = r'([a-zA-Z\s]*)\((.*)\)\s*=\s*\{(.*)\}\;.*'
    p = re.compile(pattern)
    matches = p.findall(geo)

    data = {}
    if len(matches) == 0:
        return None
    
    out = {}

    for m in matches:
        data = m[2].strip().split(', ')
        out = {'type' : m[0], 'id' :  m[1], 'data' : data}

    return out

def get_variables(geo):
    """
    Finds all gmsh variables in a .geo script. 
    Variables use the gmsh syntax `variable = value;` 

    Parameters

    geo : string,
          The .geo script to search for variables in

    Returns

    out : dict,
          The key is the variable name and the value is
          the value assigned to the variable.
          If no variables are found this function returns
          `None`.

    """
    import re 
    pattern = '(^[a-zA-Z_]+\w*)\s*=\s*(.*);'
    p = re.compile(pattern, re.M)
    matches = p.findall(geo)

    if len(matches) == 0:
        return None

    out = {}
    for m in matches:
        out[m[0]] = m[1]

    return out 

def check_objmembers(obj, members):
    """
    Determines if all the parameters in one or more objs exist in a list containing
    the parameter IDs.

    Parameters

    obj : dict,
            A obj is an item in the dictionary where the key in the dictionary
            is the obj ID and the value are the obj members that are specified 
            as a list.

    Returns 

    out : bool,
          If all of the items in `members` are found in any obj of
          `objs` then this function returns True, and False otherwise.


    # Examples

    >>> points = {'p0' : [0.0, 0.0], 'p1' : [1.0, 2.0], 'p2' : [2.0, 3.0] }
    >>> splines = {'0' : ['p0', 'p1', 'p2'], '1' : ['p1', 'p2'] }
    >>> check_objmembers(splines, points)
    True

    """

    from six import itervalues
    out = True
    for val in itervalues(obj):
        for v in val:
            if not v in members:
                out = False
    return out

def eval_vars(variables):
    """
    Evaluates variables that get their values
    from counters

    Parameters

    variables : dict,
                contains the variables to evaluate

    Returns : dict,
               variables with updated values.

    """

    from six import iteritems

    out = { }
    for k, v in iteritems(variables):
        if v in counter.counter:
            out[k] = counter[v]
        else:
            out[k] = variables[k]
    return out

def eval_id(idx, variables):
    """
    Evaluates the id string of a gmsh object (Point, Spline ... ).
    If the string contains any reference to a variable, this
    reference is replaced by the variable value

    Parameters 

    idx : string,
          the ID string to evaluate 
    variables : dict,
                the key is the variable and the value is the value of the variable

    Returns 

    out : int,
          the ID obtained after evaluation.

    """

    from six import iteritems
    import re

    for k, v in iteritems(variables):
      idx = re.sub(k, str(v), str(idx))

    return int(eval(idx))

def eval_data(obj, variables):
    """
    Evaluates the data field of a gmsh object (Point, Spline, ...).
    If any of the data items of the object contains a reference to 
    a variable, then the reference is replaced by the variable value.


    Parameters

    obj : dict,
          The object to evaluate. Must contain obj['data'] and obj['type']
    variables : dict,
                the key is the variable and the value is the value of the variable
            

    Returns 

    out : list,
          The evaluated object data. The type of the items in the list depend on the 
          type of the object data. 

    """
    
    from six import iteritems
    import re
    
    data = []
    for d in obj['data']:
      d_ = d
      for k, v in iteritems(variables):
          d_ = re.sub(k, str(v), str(d_))
      try:
        data.append(datatypes[obj['type']](eval(d_))) 
      except:
        data.append(d_)

    return data

def write(filename, var, cmd, kernel='OpenCascade'):
    """
    Writes gmsh variables and objects to geo script. 
    
    Parameters

    filename : string,
               Path and name including extension to geo the script to write.
               If this file does not exist, it will be created. If it already exists
               it is overwritten.

        var  : dict,
               collection of gmsh variables, where
               key is the variable name and value is the variable value.

    objects : dict,
              collection of gmsh objects, where
              key is object type and value is the object.
              Each object is a dict that has the keys `id`, `data`, and `type`.

    kernel : string, optional,
             geometry kernel to use.
             Make sure to use the 'OpenCascade' geometry kernel to 
             be able to export the geometry to any of the OpenCascade formats 
             (brep or step).
             


    """

    from six import iterkeys

    f = open(filename, 'w')
    
    if kernel == 'OpenCascade':
        f.write('SetFactory("OpenCASCADE");\n')

    f.write(write_variables(var))

    for k in supportedobjs:
        if k in cmd:
            f.write(write_object(k, cmd[k]))

    f.close()

def read(filename):
    """
    Reads a gmsh geo script file.

    Parameters 

    filename : string,
               Path and filename to the geo script to read

    Returns

    variables : dict,
                collection of gmsh variables, where
                key is the variable name and value is the variable value.
    objects : dict,
              collection of gmsh objects, where
              key is object type and value is the object.
              Each object is a dict that has the keys `id`, `data`, and `type`.

    """

    from six import iteritems

    f = open(filename, 'r')

    pts = []

    counter.reset()
    variables = {}
    objects = {}

    for k in supportedobjs:
      objects[k] = {}

    for line in f:
        # Get variables
        var = get_variables(line)
        if var:
            var = eval_vars(var)
            for k, v in iteritems(var):
              variables[k] = v
        # Get objects
        obj = get_object(line)
        if obj:
          obj['id'] = eval_id(obj['id'], variables)
          obj['data'] = eval_data(obj, variables)
          if obj['type'] in counters:
            count = counter[counters[obj['type']]]
          objects[obj['type']][obj['id']] = obj['data'] 
    return variables, objects
                
def write_object(cmd, obj):
    """
    Writes a gmsh object to string.

    Parameters

    cmd : string,
          gmsh object to write
    obj : dict, or list,
            contains the objs and their parameters to write

    Returns

    out : string,
          contains the gmsh object. Each object is placed on a new line. 

    """
    
    from six import iteritems

    out = []
    if isinstance(obj, dict):
        for k, v in iteritems(obj):
            out.append(_write_object(cmd, k, v))
    else:
        for k, v in enumerate(obj):
            out.append(_write_object(cmd, k, v))

    return ''.join(out)

def write_variables(var):
    """
    Writes gmsh variables to string.

    Parameters

    var : dict,
          contains the variables to write 

    Returns

    out : string,
          the variables written. Each variable is placed on a new line. 

    """
    
    from six import iteritems

    out = []
    for k, v in iteritems(var):
        out.append('%s = %s;\n' % (k, str(v)))

    return ''.join(out)

def _write_object(cmd, k, v):
    out = '%s(%s) = {' % (cmd, str(k))
    out += ', '.join(map(lambda vi : str(vi), v))
    out += '};\n'
    return out

