from os.path import join, dirname

def get_path_pandapower_json_test_data() -> str:
    path = dirname(__file__)
    path = join(path, "json")
    path = join(path, "pandapower_test_conf.json")
    return path

def get_path_case9_mat() -> str:
    path = dirname(__file__)
    path = join(path, "matpower")
    path = join(path, "case9.mat")
    return path