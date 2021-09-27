from os.path import join, dirname

def get_path_pandapower_json_test_data() -> str:
    path = dirname(__file__)
    path = join(path, "json")
    path = join(path, "pandapower_test_conf.json")
    return path

def get_path_pandapower_json_test_excel_profiles_data() -> str:
    path = dirname(__file__)
    path = join(path, "json")
    path = join(path, "pandapower_test_conf_excel_profiles.json")
    return path

def get_excel_timeseries() -> str:
    path = dirname(__file__)
    path = join(path, "excel")
    path = join(path, "time_series_jsonreader.xlsx")
    return path

def get_path_case9_mat() -> str:
    path = dirname(__file__)
    path = join(path, "matpower")
    path = join(path, "case9.mat")
    return path

def get_clustering_test_data() -> str:
    path = dirname(__file__)
    path = join(path, "excel")
    path = join(path, "normalized_demand_profiles.xlsx")
    return path

def get_clustering_data_test() -> str:
    path = dirname(__file__)
    path = join(path, "excel")
    path = join(path, "data_clusters_tests.xlsx")
    return path

def set_pandapower_test_output_directory() -> str:
    path = dirname(__file__)
    path = join(path, "outputs")
    return path