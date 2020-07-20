import json


def get_args_list(args):
    """
    Create a list in order of key, value for command line args.
    For no value args in .json file use value of [].
    :sys.argv args: command line args
    """
    args_list = []
    file_path = args[3]

    with open(file_path) as f:
        data = json.load(f)

    for k, v in data.items():
        args_list.append(str(k))
        if type(v) is list and len(v) == 0:
            continue
        args_list.append(str(v))

    return args_list, file_path
