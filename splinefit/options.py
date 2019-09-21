def get_options(args, delimiter="="):
    d_args = {}
    for arg in args[1::]:
        key, value = arg.split(delimiter)
        d_args[key] = value
    return d_args

def check_options(args, options):
    args = get_options(args[1:])
    for arg in args:
        if "--" in arg:
            _arg = arg[2:]
        else:
            _arg = arg

        if _arg not in options:
            raise ValueError("Unknown option: %s" %  _arg)

def print_options(options):
    for option in options:
        print(option, ":", options[option])
