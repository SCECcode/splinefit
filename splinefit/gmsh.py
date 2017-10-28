supportedcmds = {'Spline', 'Point', 'Line Loop', 'Ruled Surface'} 
datatypes = {'Spline' : lambda x : map(lambda y: int(y), x), 
             'Point'  : lambda x : map(lambda y: float(y), x)}
counters = {'newp', 'newl', 'news', 'newv', 'newll'}


class Counter:

    def __init__(self):
        self.counter = {}
        for c in counters:
            self.counter[c] = 0

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
    p = re.compile(pattern)
    matches = p.findall(geo)

    if len(matches) == 0:
        return None

    out = {}
    for m in matches:
        out[m[0]] = m[1]

    return out 

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
        try: 
            out[k] = datatypes[grouptype](v)
        except:
            out[k] = groups[k]
    return out

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
    for gk, gv in iteritems(group):
        out[gk] = []
        for gvi, gvv in enumerate(gv):
            out[gk].append(gvv)

    for vk, vv in iteritems(values):
        for gk, gv in iteritems(group):
            for gvi, gvv in enumerate(gv):
                updated_value = re.sub(vk, str(vv), str(gvv))
                if updated_value != str(gvv):
                    out[gk][gvi] = updated_value
    return out
                
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

    out = ''
    if isinstance(group, dict):
        for k, v in iteritems(group):
            out += _write_command(cmd, k, v)
    else:
        for k, v in enumerate(group):
            out += _write_command(cmd, k, v)

    return out

def _write_command(cmd, k, v):
    out = '%s(%d) = {' % (cmd, int(k))
    out += ', '.join(map(lambda vi : str(vi), v))
    out += '};\n'
    return out


                
    
