supportedcmds = {'Spline', 'Point', 'Line Loop', 'Ruled Surface'} 
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

    group : dictionary,
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

def eval(variables):
    from six import iteritems

    out = { }
    for k, v in iteritems(variables):
        if v in counter.counter:
            out[k] = counter[v]
        else:
            out[k] = variables[k]
    return out

