supportedcmds = ['Point', 'Spline', 'Line Loop', 'Ruled Surface', 'Surface']
datatypes = {'Spline' : lambda x : int(x), 
             'Point'  : lambda x : float(x), 
             'Ruled Surface' : lambda x : int(x),
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


def get_command(cmd, geo):
    """
    Searches through a gmsh geo script for a specific command and extracts
    all occurences found. 
    A gmsh command uses the syntax: `command(id) = {.., .., parameters, etc};`

    Parameters

    cmd : string,
          The command to search for
    geo : string,
          The input geo-script to parse. 

    Returns

    out : dict,
          All of the instances found of the command. 
          The command id is used as the key and a list of parameters
          is the value. If no instance of `cmd` is found, then this function
          returns `None`.

    # Examples

    >>> geo = "Point(1) = {1.0, 2.0, 3.0};"
    >>> point = get_command('Point', geo)
    >>> point
    {'1': ['1.0', '2.0', '3.0']}

    """

    import re 

    if cmd not in supportedcmds:
        raise KeyError("Command: %s not supported" %cmd)

    pattern = r'%s\((.*)\)\s*=\s*\{(.*)\}\;.*'%cmd
    p = re.compile(pattern)
    matches = p.findall(geo)

    data = {}
    if len(matches) == 0:
        return None
    
    out = {}

    for m in matches:
        data = m[1].strip().split(', ')
        out[m[0]] = data

    return out

def get_group(geo):

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

def read(filename):
    from six import iterkeys, iteritems
    pass

    geo = open(filename).read() 

    var = get_variables(geo)

    cmd = {}
    for k in supportedcmds:
        cmd[k] = get_command(k, geo)
        check_groupmembers(cmd[k], var)

    for k in iterkeys(var):
        var = eval_vars(var)
    
    for k in iterkeys(var):
        var = eval_vars(var)

    for k in iterkeys(cmd):
        cmd[k] = subs(cmd[k], var) 

    for k in iterkeys(cmd):
        cmd[k] = eval_groups(cmd[k], k)

    return var, cmd

def check_groupmembers(group, members):
    """
    Determines if all the parameters in one or more groups exist in a list containing
    the parameter IDs.

    Parameters

    group : dict,
            A group is an item in the dictionary where the key in the dictionary
            is the group ID and the value are the group members that are specified 
            as a list.

    Returns 

    out : bool,
          If all of the items in `members` are found in any group of
          `groups` then this function returns True, and False otherwise.


    # Examples

    >>> points = {'p0' : [0.0, 0.0], 'p1' : [1.0, 2.0], 'p2' : [2.0, 3.0] }
    >>> splines = {'0' : ['p0', 'p1', 'p2'], '1' : ['p1', 'p2'] }
    >>> check_groupmembers(splines, points)
    True

    """

    from six import itervalues
    out = True
    for val in itervalues(group):
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

def eval_groups(groups, grouptype):
    """
    Evaluates groups by converting their parameters
    to their particular type. If the type conversion fails, 
    then the original value is reused.

    Parameters

    variables : dict,
                contains the groups to evaluate

    Returns  : dict,
               groups with updated values.

    """
    from six import iteritems

    out = { }
    for k, v in iteritems(groups):
        data = []
        for vi in v:
            try: 
                data.append(datatypes[grouptype](vi))
            except:
                data.append(vi)
        out[k] = data
    return out


def eval_id(idx, variables):

    from six import iteritems
    import re

    for k, v in iteritems(variables):
      idx = re.sub(k, str(v), str(idx))

    return int(eval(idx))

def eval_data(group, variables):
    
    from six import iteritems
    import re
    
    data = []
    for d in group['data']:
      d_ = d
      for k, v in iteritems(variables):
          d_ = re.sub(k, str(v), str(d_))
      try:
        data.append(datatypes[group['type']](eval(d_))) 
      except:
        data.append(d_)

    return data


def subs(group, values):
    """
    Substitutes group parameters using a dictionary of key-value
    pairs corresponding to the parameter to perform substitution
    for and its new value.

    Parameters

    group : dict,
            contains the groups to perform substition for

    values : dict,
             contains the new parameter values.

    Returns

    out : dict,
          contains the updated values. If no parameters to
          update are found, then the original dict `group` is
          returned.
    """

    import re
    from six import iteritems

    out = {}
    for vk, vv in iteritems(values):
        for gk, gv in iteritems(group):
            out[gk] = []
            newgk = re.sub(vk, str(vv), str(gk))
            for gvi, gvv in enumerate(gv):
                out[newgk].append(gvv)

    for vk, vv in iteritems(values):
        for gk, gv in iteritems(group):
            # Update key
            newgk = re.sub(vk, str(vv), str(gk))
            out[newgk] = []
            #if updated_value != str(gk):
            #    newgk = updated_value
            #else:
            #    newgk = gk

            # Update value
            for gvi, gvv in enumerate(gv):
                updated_value = re.sub(vk, str(vv), str(gvv))
                if updated_value != str(gvv):
                    out[newgk].append(updated_value)
                else:
                    out[newgk].append(gvv)
    return out

def write(filename, var, cmd):

    from six import iterkeys

    f = open(filename, 'w')

    f.write(write_variables(var))
    for k in supportedcmds:
        if k in cmd:
            f.write(write_command(k, cmd[k]))

    f.close()
                
def write_command(cmd, group):
    """
    Writes a gmsh command to string.

    Parameters

    cmd : string,
          gmsh command to write
    group : dict, or list,
            contains the groups and their parameters to write

    Returns

    out : string,
          contains the gmsh commands. Each command is placed on a new line. 

    """
    
    from six import iteritems

    out = []
    if isinstance(group, dict):
        for k, v in iteritems(group):
            out.append(_write_command(cmd, k, v))
    else:
        for k, v in enumerate(group):
            out.append(_write_command(cmd, k, v))

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

def _write_command(cmd, k, v):
    out = '%s(%s) = {' % (cmd, str(k))
    out += ', '.join(map(lambda vi : str(vi), v))
    out += '};\n'
    return out

def parse(filename):

    from six import iteritems

    f = open(filename, 'r')

    pts = []

    counter.reset()
    variables = {}
    groups = {}

    for k in supportedcmds:
      groups[k] = {}

    for line in f:
        var = get_variables(line)
        if var:
            bvar = var
            var = eval_vars(var)
            for k, v in iteritems(var):
              variables[k] = v
        grp = get_group(line)
        if grp:
          grp['id'] = eval_id(grp['id'], variables)
          grp['data'] = eval_data(grp, variables)
          if grp['type'] in counters:
            count = counter[counters[grp['type']]]
          groups[grp['type']][grp['id']] = grp['data'] 
    return variables, groups
