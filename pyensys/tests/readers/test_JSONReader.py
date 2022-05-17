from math import sqrt
from os.path import join, dirname
from pandas import Timestamp, DataFrame

from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.readers.JSONReader import _create_dataframe, _load_pandapower_profile_data, _load_output_variable, \
    _load_output_variables, _read_file, _read_dataframe_data, _load_optimisation_profile_data, ReadJSON
from pyensys.readers.ReaderDataClasses import DataframeData
from pyensys.tests.test_data_paths import get_path_pandapower_json_test_data, \
    get_excel_timeseries, get_clustering_test_data


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
            "problem_optimizer": "recursive_function",
            "opf_type": "ac"
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
    assert problem_settings.opf_type == "ac"


def test_load_problem_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    problem_settings = power_system._load_problem_settings()
    assert not problem_settings.initialised


def test_load_opf_time_settings_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "opf_time_settings": {
            "begin": "2021-1-1 00:00:00",
            "end": "2021-1-1 02:00:00",
            "frequency": "hourly",
            "time_block": "1"
        }
    }
    date_time_settings = power_system._load_opf_time_settings()
    assert date_time_settings.date_time_settings.tolist() == \
           [Timestamp('2021-01-01 00:00:00', freq='H'),
            Timestamp('2021-01-01 01:00:00', freq='H'),
            Timestamp('2021-01-01 02:00:00', freq='H')]
    assert date_time_settings.initialised


def test_load_opf_time_settings_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    date_time_settings = power_system._load_opf_time_settings()
    assert not date_time_settings.initialised


def test_load_pandapower_mpc_setting_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "pandapower_mpc_settings": {
            "mat_file_path": "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\matpower\\case9.mat",
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
                                   column_names=["load1_p"])
    RESULTS = _create_dataframe(dataframe_data)
    assert DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=["load1_p"]).equals(RESULTS)


def test_load_pandapower_profile_data():
    profile_data_dict = {
        "1": {
            "dataframe_columns_names": ["load1_p"],
            "data": [[67.28095505], [9.65466896], [11.70181664]],
            "element_type": "load",
            "indexes": [0],
            "variable_name": "p_mw",
            "active_columns_names": ["load1_p"],
            "format_data": "original"
        }
    }
    profile_data = _load_pandapower_profile_data(profile_data_dict["1"])
    assert profile_data[0].indexes == [0]
    assert profile_data[0].variable_name == "p_mw"
    assert profile_data[0].element_type == "load"
    assert profile_data[0].active_columns_names == ["load1_p"]


def test_load_pandapower_profiles_data_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "pandapower_profiles_data": {
            "1": {
                "dataframe_columns_names": ["load1_p"],
                "data": [[67.28095505], [9.65466896], [11.70181664]],
                "element_type": "load",
                "indexes": [0],
                "variable_name": "p_mw",
                "active_columns_names": ["load1_p"],
                "format_data": "original"
            },
            "2": {
                "dataframe_columns_names": ["gen1_p"],
                "data": [[67.28095505], [9.65466896], [11.70181664]],
                "element_type": "gen",
                "indexes": [0],
                "variable_name": "p_mw",
                "active_columns_names": ["gen1_p"],
                "format_data": "original"
            }
        }
    }
    data = power_system._load_pandapower_profiles_data()
    assert len(data.data) == 2
    assert data.initialised


def test_load_pandapower_profiles_data_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    data = power_system._load_pandapower_profiles_data()
    assert not data.initialised


def test_load_output_variable():
    output_variable_dict = {
        "name_dataset": "res_load",
        "name_variable": "p_mw",
        "variable_indexes": [0]
    }
    output_variable = _load_output_variable(output_variable_dict)
    assert output_variable.name_dataset == "res_load"
    assert output_variable.name_variable == "p_mw"
    assert output_variable.variable_indexes == [0]


def test_load_output_variables():
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
    output_variables = _load_output_variables(output_variables_dict)
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
    assert settings.directory == "C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\outputs"
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
    assert p.opf_time_settings.initialised
    assert p.output_settings.initialised
    assert p.pandapower_mpc_settings.initialised
    assert p.pandapower_optimisation_settings.initialised
    assert p.problem_settings.initialised
    assert p.pandapower_profiles_data.initialised
    assert p.initialised


def test_read_file():
    path = get_excel_timeseries()
    profile_data_dict = {"excel_sheet_name": "LoadP"}
    RESULT = _read_file(path, profile_data_dict)
    assert DataFrame(data=[[0.089588, 0.294746], [0.089359, 0.293991], [0.087981, 0.289459]],
                     columns=["B1", "B2"]).equals(RESULT)


def test_read_dataframe_data_case1():
    profile_data_dict = {
        "dataframe_columns_names": ["load1_p"],
        "data": [[67.28095505], [9.65466896], [11.70181664]],
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]],
                     columns=["load1_p"]).equals(RESULT)


def test_read_dataframe_data_case2():
    profile_data_dict = {
        "data": [[67.28095505], [9.65466896], [11.70181664]],
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert RESULT.empty


def test_read_dataframe_data_case3():
    profile_data_dict = {
        "dataframe_columns_names": ["load1_p"]
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert RESULT.empty


def test_read_dataframe_data_case4():
    profile_data_dict = {
        "data_path": get_excel_timeseries(),
        "excel_sheet_name": "LoadP",
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert DataFrame(data=[[0.089588, 0.294746], [0.089359, 0.293991], [0.087981, 0.289459]],
                     columns=["B1", "B2"]).equals(RESULT)


def test_read_dataframe_data_case5():
    profile_data_dict = {
        "data_path": get_excel_timeseries()
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert RESULT.empty


def test_read_dataframe_data_case6():
    profile_data_dict = {
        "excel_sheet_name": "LoadP"
    }
    RESULT = _read_dataframe_data(profile_data_dict)
    assert RESULT.empty


def test_load_optimisation_profile_data():
    profile_data_dict = {
        "data_path": get_clustering_test_data(),
        "excel_sheet_name": "Sheet1",
        "element_type": "load",
        "variable_name": "p_mw"
    }
    profile_data = _load_optimisation_profile_data(profile_data_dict)
    assert profile_data.variable_name == "p_mw"
    assert profile_data.element_type == "load"
    assert profile_data.data.shape == (46, 8)


def test_load_optimisation_profiles_data_case1():
    power_system = ReadJSON()
    power_system.settings = {
        "optimisation_profiles_data": {
            "1": {
                "data_path": get_clustering_test_data(),
                "excel_sheet_name": "Sheet1",
                "element_type": "load",
                "variable_name": "p_mw"
            }
        }
    }
    data = power_system._load_optimisation_profiles_data()
    assert len(data.data) == 1
    assert data.initialised


def test_load_optimisation_profiles_data_case2():
    power_system = ReadJSON()
    power_system.settings = {}
    data = power_system._load_optimisation_profiles_data()
    assert not data.initialised


def test_load_optimisation_binary_variables():
    power_system = ReadJSON()
    power_system.settings = {
        "optimisation_binary_variables": [
            {
                "element_type": "gen",
                "variable_name": "installation",
                "elements_ids": ["G0"],
                "costs": [1.0],
                "elements_positions": [3]
            },
            {
                "element_type": "AC line",
                "variable_name": "installation",
                "elements_ids": ["L1"],
                "costs": [2.0],
                "elements_positions": [1]
            }
        ]
    }
    power_system._load_optimisation_binary_variables()
    assert len(power_system.parameters.optimisation_binary_variables) == 2
    assert power_system.parameters.optimisation_binary_variables[0].element_type == "gen"
    assert power_system.parameters.optimisation_binary_variables[1].element_type == "AC line"
    assert power_system.parameters.optimisation_binary_variables[1].variable_name == "installation"
    assert power_system.parameters.optimisation_binary_variables[0].elements_ids == ["G0"]
    assert power_system.parameters.optimisation_binary_variables[0].elements_positions == [3]
    assert power_system.parameters.optimisation_binary_variables[1].costs == [2.0]


def test_load_problem_settings_case3():
    power_system = ReadJSON()
    power_system.settings = {
        "problem": {
            "return_rate_in_percentage": 3.0
        }
    }
    problem_settings = power_system._load_problem_settings()
    assert problem_settings.return_rate_in_percentage == 3.0


def test_read_pandapower_profiles_with_attest_format_from_excel():
    settings = {
        "1": {
            "data_path": join(dirname(__file__), "..", "excel", "time_series_pandapower_attest_format.xlsx"),
            "excel_sheet_name": "Sheet1",
            "element_type": "load",
            "format_data": "attest"
        }
    }
    profiles_data = _load_pandapower_profile_data(settings["1"])
    assert len(profiles_data) == 2
    assert DataFrame(data=[[1, 5], [2, 6]], columns=["pd_0", "pd_2"], index=[2, 3]).equals(profiles_data[0].data)
    assert DataFrame(data=[[3, 7], [4, 8]], columns=["qd_0", "qd_2"], index=[2, 3]).equals(profiles_data[1].data)
    assert profiles_data[0].element_type == "load"
    assert profiles_data[1].element_type == "load"
    assert profiles_data[0].variable_name == "p_mw"
    assert profiles_data[1].variable_name == "q_mvar"
    assert profiles_data[0].indexes == [0, 2]
    assert profiles_data[1].indexes == [0, 2]
    assert profiles_data[0].active_columns_names == ["pd_0", "pd_2"]
    assert profiles_data[1].active_columns_names == ["qd_0", "qd_2"]


def test_adjust_pandapower_profiles_to_time_settings():
    readjs = ReadJSON()
    readjs.settings = {
        "opf_time_settings": {
            "begin": "00:00:00",
            "end": "01:00:00",
            "frequency": "hourly",
            "time_block": "1"
        },
        "pandapower_profiles_data": {
            "1": {
                    "data_path": join(dirname(__file__), "..", "excel", "time_series_pandapower_attest_format.xlsx"),
                    "excel_sheet_name": "Sheet2",
                    "element_type": "load",
                    "format_data": "attest"
            }
        }
    }
    readjs.parameters.opf_time_settings = readjs._load_opf_time_settings()
    readjs.parameters.pandapower_profiles_data = readjs._load_pandapower_profiles_data()
    readjs._adjust_pandapower_profiles_to_time_settings()
    assert DataFrame(data=[[3.0, 7.0], [4.0, 8.0]], columns=["pd_0", "pd_2"]).equals(
        readjs.parameters.pandapower_profiles_data.data[0].data)
    assert DataFrame(data=[[5.0, 9.0], [6.0, 10.0]], columns=["qd_0", "qd_2"]).equals(
        readjs.parameters.pandapower_profiles_data.data[1].data)


def test_store_profiles_dataframes_in_load_optimisation_profiles_data():
    settings = {
        "optimisation_profiles_data": {
            "format_data": "attest",
            "data": [
                {
                    "group": "a",
                    "data": [
                        [1.0]
                    ],
                    "columns_names": ["a"]
                },
                {
                    "group": "b",
                    "data": [
                        [2.0]
                    ],
                    "columns_names": ["b"]
                }
            ]
        }
    }
    reader = ReadJSON()
    reader.settings = settings
    reader._load_optimisation_profiles_data()
    assert len(reader.parameters.optimisation_profiles_dataframes) == 2
    assert DataFrame(data=[[1.0]], columns=["a"]).equals(reader.parameters.optimisation_profiles_dataframes["a"])
    assert DataFrame(data=[[2.0]], columns=["b"]).equals(reader.parameters.optimisation_profiles_dataframes["b"])


def test_calculate_total_apparent_power_per_scenario_and_per_year_for_optimisation_profiles():
    settings = {
        "optimisation_profiles_data":{
		"format_data": "attest",
		"data": [
			{
				"group": "buses",
				"data": [
					[1, 2030, 0, 12.02, 7.03],
					[1, 2030, 2, 15.01, 10.04],
					[2, 2030, 0, 11.98, 6.97],
					[2, 2030, 2, 14.99, 9.96]
				],
				"columns_names": ["scenario", "year", "bus_index", "p_mw", "q_mvar"]
			}
		]
    }
    }
    reader = ReadJSON()
    reader.settings = settings
    expected = DataFrame(data=[[1, 2030, sqrt(12.02**2 + 7.03**2)+sqrt(15.01**2 + 10.04**2)],
                               [2, 2030, sqrt(11.98**2 + 6.97**2)+sqrt(14.99**2 + 9.96**2)]])
    assert len(reader.parameters.optimisation_profiles_dataframes) == 2
    assert DataFrame(data=[[1.0]], columns=["a"]).equals(reader.parameters.optimisation_profiles_dataframes["a"])
    assert DataFrame(data=[[2.0]], columns=["b"]).equals(reader.parameters.optimisation_profiles_dataframes["b"])
