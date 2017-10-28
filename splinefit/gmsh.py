# Spline\((\d+)\)\s*=\s*\{2,(\sp\d+\s\+\s\d+,)

supportedcmds = {'Spline', 'Point', 'Line Loop','Ruled Surface'} 

class Command:

    def __init__(self, cmd, data):

        self.cmd = cmd

def get_command(cmd, geo):
    import re 

    if cmd not in supportedcmds:
        raise KeyError("Command: %s not supported" %cmd)

    pattern = '%s\((.*)\)\s*=\s*\{(.*)\};'%cmd
    p = re.compile(pattern)

    matches = p.findall(geo, re.M)

    data = {}
    if len(matches) == 0:
        return None
    
    cmd = {}

    for m in matches:
        data = m[1].strip().split(', ')
        cmd[m[0]] = data

    return cmd


def get_variables(var, geo):
    pass


def check_groupmembers(group, members):
    from six import itervalues
    for val in itervalues(group):
        for v in val:
            assert v in members
