import json


def get_args_list(args, pos):
    """
    Create a list in order of key, value for command line args.
    For no value args in .json file use value of [].
    :list args: copy of sys.argv command line args
    :int pos: position in list of the .json file
    """
    args_list = []
    file_path = args[pos]

    with open(file_path) as f:
        data = json.load(f)

    for k, v in data.items():
        # keys must use "--" to start but this is optional for user input
        if k[:2] != "--":
            k = _fix_key_format(k)

        args_list.append(str(k))
        if type(v) is list and len(v) == 0:
            continue
        args_list.append(str(v))

    return _set_args(args, args_list, file_path)


def _set_args(args, line_args, file_name):
    """
    Sets the arg list to contain all of the original CL
    arguments and the arguments provided in the file.
    :list args: copy of sys.argv command line args
    :list args_list: ordered list of key, value args from file
    :str file_name: name file to remove from args list
    """
    args = args + line_args
    args.remove(file_name)

    return args


def _fix_key_format(key):
    """
    If the key for the argument does not start with "--"
    this function recurses until it fixes this.
    :str key: the key of the argument in question
    """
    if key[:2] == "--":
        return key
    n_key = "-" + key

    return _fix_key_format(n_key)
