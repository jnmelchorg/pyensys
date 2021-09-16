from os import read
from pandas.core.frame import DataFrame
from pyensys.readers.JSONReader import ReadJSON, DataframeData
from pandas import Timestamp
from pyensys.tests.tests_data_paths import get_path_pandapower_json_test_data

def test_load_problem_settings_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "problem": {
            "system": "distribution",
            "name": "planning",
            "multi_objective": True,
            "stochastic": True,
            "intertemporal": True,
            "opf_optimizer": "pandapower",
            "problem_optimizer": "recursive_function"
        }
    }
    problem_settings = power_system._load_problem_settings()
    assert problem_settings.problem == "planning"
    assert problem_settings.multi_objective
    assert problem_settings.stochastic
    assert problem_settings.intertemporal
    assert problem_settings.initialised
    assert problem_settings.opf_optimizer == "pandapower"
    assert problem_settings.problem_optimizer == "recursive_function"

def test_load_problem_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    problem_settings = power_system._load_problem_settings()
    assert not problem_settings.initialised

def test_load_time_related_settings_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "time_related_settings": {
            "begin": "2021-1-1 00:00:00",
            "end": "2021-1-1 02:00:00",
            "frequency": "hourly",
            "time_block": "1"
        }
    }
    date_time_settings = power_system._load_time_series_settings()
    assert date_time_settings.date_time_settings.tolist() == \
        [Timestamp('2021-01-01 00:00:00', freq='H'), 
        Timestamp('2021-01-01 01:00:00', freq='H'), 
        Timestamp('2021-01-01 02:00:00', freq='H')]
    assert date_time_settings.initialised

def test_load_time_related_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    date_time_settings = power_system._load_time_series_settings()
    assert not date_time_settings.initialised

def test_load_pandapower_mpc_setting_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "pandapower_mpc_settings": {
            "mat_file_path" : "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\matpower\\case9.mat",
            "frequency": 60.0
        }
    }
    mpc_settings = power_system._load_pandapower_mpc_settings()
    assert mpc_settings.mat_file_path == \
        "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\matpower\\case9.mat"
    assert mpc_settings.system_frequency == 60.0
    assert mpc_settings.initialised

def test_load_pandapower_mpc_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    mpc_settings = power_system._load_pandapower_mpc_settings()
    assert not mpc_settings.initialised

def test_create_dataframe():
    dataframe_data = DataframeData(data=[[67.28095505], [9.65466896], [11.70181664]], 
        column_names=["load1_p"], row_names=["2021-01-01 00:00:00", "2021-01-01 01:00:00", 
        "2021-01-01 02:00:00"])
    power_system = ReadJSON()
    RESULTS = power_system._create_dataframe(dataframe_data)
    assert  DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
            index=["2021-01-01 00:00:00", "2021-01-01 01:00:00", 
        "2021-01-01 02:00:00"], columns=["load1_p"]).equals(RESULTS)
    
def test_load_profile_data():
    power_system = ReadJSON()
    profile_data_dict = {
        "1": {
            "dataframe_columns_names": ["load1_p"],
            "data":[[67.28095505], [9.65466896], [11.70181664]],
            "dataframe_rows_date_time": ["2021-01-01 00:00:00", "2021-01-01 01:00:00", 
                "2021-01-01 02:00:00"],
            "element_type": "load",
            "indexes": [0],
            "variable_name": "p_mw"
        }
    }
    profile_data = power_system._load_profile_data(profile_data_dict["1"])
    assert profile_data.indexes == [0]
    assert profile_data.variable_name == "p_mw"
    assert profile_data.element_type == "load"

def test_load_profiles_data_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "profiles_data": {
            "1": {
                "dataframe_columns_names": ["load1_p"],
                "data":[[67.28095505], [9.65466896], [11.70181664]],
                "dataframe_rows_date_time": ["2021-01-01 00:00:00", "2021-01-01 01:00:00", 
                "2021-01-01 02:00:00"],
                "element_type": "load",
                "indexes": [0],
                "variable_name": "p_mw"
            },
            "2": {
                "dataframe_columns_names": ["gen1_p"],
                "data":[[67.28095505], [9.65466896], [11.70181664]],
                "dataframe_rows_date_time": ["2021-01-01 00:00:00", "2021-01-01 01:00:00", 
                "2021-01-01 02:00:00"],
                "element_type": "gen",
                "indexes": [0],
                "variable_name": "p_mw"
            }
        }
    }
    data = power_system._load_profiles_data()
    assert len(data.data) == 2
    assert data.initialised

def test_load_profiles_data_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    data = power_system._load_profiles_data()
    assert not data.initialised

def test_load_output_variable():
    power_system = ReadJSON()
    output_variable_dict = {
        "name_dataset": "res_load",
        "name_variable": "p_mw",
        "variable_indexes": [0]
    }
    output_variable = power_system._load_output_variable(output_variable_dict)
    assert output_variable.name_dataset == "res_load"
    assert output_variable.name_variable == "p_mw"
    assert output_variable.variable_indexes == [0]

def test_load_output_variables():
    power_system = ReadJSON()
    output_variables_dict = {
        "1": {
            "name_dataset": "res_load",
            "name_variable": "p_mw",
            "variable_indexes": []
        },
        "2": {
            "name_dataset": "res_bus",
            "name_variable": "vm_pu",
            "variable_indexes": []
        }         
    }
    output_variables = power_system._load_output_variables(output_variables_dict)
    assert len(output_variables) == 2

def test_load_output_settings_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "output_settings": {
            "directory": "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\outputs",
            "format": ".xlsx",
            "output_variables": {
                "1": {
                    "name_dataset": "res_load",
                    "name_variable": "p_mw",
                    "variable_indexes": []
                },
                "2": {
                    "name_dataset": "res_bus",
                    "name_variable": "vm_pu",
                    "variable_indexes": []
                }
            }
        }           
    }
    settings = power_system._load_output_settings()
    assert settings.directory ==  "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\outputs"
    assert settings.format == ".xlsx"
    assert len(settings.output_variables) == 2
    assert settings.initialised

def test_load_output_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    settings = power_system._load_output_settings()
    assert not settings.initialised

def test_load_pandapower_optimisation_settings_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "pandapower_optimisation_settings": {
            "display_progress_bar": False,
            "continue_on_divergence": False,
            "optimisation_software": "pypower"
        }
    }
    settings = power_system._load_pandapower_optimisation_settings()
    assert not settings.display_progress_bar
    assert not settings.continue_on_divergence
    assert settings.optimisation_software == "pypower"
    assert settings.initialised

def test_load_pandapower_optimisation_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    settings = power_system._load_pandapower_optimisation_settings()
    assert not settings.initialised

def test_load_parameters_power_system_optimisation():
    file_path = get_path_pandapower_json_test_data()
    read_json = ReadJSON()
    read_json.read_json_data(file_path)
    p = read_json.parameters
    assert p.date_time_optimisation_settings.initialised
    assert p.output_settings.initialised
    assert p.pandapower_mpc_settings.initialised
    assert p.pandapower_optimisation_settings.initialised
    assert p.problem_settings.initialised
    assert p.profiles_data.initialised
    assert p.initialised
