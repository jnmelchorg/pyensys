from os import path
from pyensys.Optimisers.RecursiveFunction import *
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable, \
    OptimisationProfileData
from pyensys.tests.tests_data_paths import get_path_case9_mat, set_pandapower_test_output_directory, \
    get_clustering_data_test
from pyensys.Optimisers.ControlGraphsCreator import ClusterData

from pandas import DataFrame, date_range, read_excel
from math import isclose
from typing import List, Dict
from unittest.mock import MagicMock


def load_test_case() -> Parameters:
    parameters = Parameters()
    parameters.pandapower_mpc_settings.mat_file_path = get_path_case9_mat()
    parameters.pandapower_mpc_settings.system_frequency = 50.0
    parameters.pandapower_mpc_settings.initialised =  True
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
            columns=['load1_p']), active_columns_names=['load1_p']),
        PandaPowerProfileData(element_type="gen", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[240], [200], [18]], \
            columns=['gen1_p']), active_columns_names=['gen1_p'])]
    parameters.output_settings.initialised = True
    parameters.output_settings.output_variables = [OutputVariable(name_dataset='res_bus', \
        name_variable='p_mw', variable_indexes=[]), OutputVariable(name_dataset='res_line', \
        name_variable='i_ka', variable_indexes=[])]
    parameters.output_settings.directory = set_pandapower_test_output_directory()
    parameters.output_settings.format = ".xlsx"
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    parameters.problem_settings.opf_optimizer = "pandapower"
    parameters.problem_settings.intertemporal = True
    parameters.optimisation_profiles_data.data = [
        OptimisationProfileData(\
            element_type="load", variable_name="p_mw", data=read_excel(io=\
            get_clustering_data_test(), sheet_name="Sheet1")),
            OptimisationProfileData(\
            element_type="gen", variable_name="p_mw", data=read_excel(io=\
            get_clustering_data_test(), sheet_name="Sheet2"))]
    parameters.optimisation_profiles_data.initialised = True
    return parameters

def load_data_multipliers() -> Dict[int, List[ClusterData]]:
    profile_modifiers = {}
    profile_modifiers[0] = []
    profile_modifiers[0].append(ClusterData(scenarios=[1, 2, 3], centroid=0.1, level=0, \
        element_type="load", variable_name="p_mw"))
    profile_modifiers[0].append(ClusterData(scenarios=[1, 2, 3], centroid=0.2, level=0, \
        element_type="gen", variable_name="p_mw"))
    profile_modifiers[1] = []
    profile_modifiers[1].append(ClusterData(scenarios=[1, 2, 3], centroid=0.3, level=0, \
        element_type="load", variable_name="p_mw"))
    profile_modifiers[1].append(ClusterData(scenarios=[1, 2, 3], centroid=0.4, level=0, \
        element_type="gen", variable_name="p_mw"))
    return profile_modifiers

def test_operational_check():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._operational_check()
    assert RF.pp_opf.wrapper.network.OPF_converged == True
    assert isclose(3583.53647, RF.pp_opf.wrapper.network.res_cost, abs_tol=1e-4)

def test_get_profile_position_to_update_case1():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    list_multipliers = load_data_multipliers()
    PROFILE_POSITION = RF._get_profile_position_to_update(list_multipliers[1][1])
    assert PROFILE_POSITION == 1

def test_get_profile_position_to_update_case2():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    list_multipliers = load_data_multipliers()
    list_multipliers[1][1].variable_name = "q_mw"
    PROFILE_POSITION = RF._get_profile_position_to_update(list_multipliers[1][1])
    assert PROFILE_POSITION == -1

def test_create_new_pandapower_profile():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._control_graph.nodes_data = load_data_multipliers()
    RESULT = RF._create_new_pandapower_profile(RF._control_graph.nodes_data[1][1])
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT.data)

def test_create_new_pandapower_profiles():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._control_graph.nodes_data = load_data_multipliers()
    RF._node_under_analysis = 1
    RESULT = RF._create_new_pandapower_profiles()
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT.data[1].data)

def test_update_pandapower_controllers():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._control_graph.nodes_data = load_data_multipliers()
    RF._node_under_analysis = 1
    RF._update_pandapower_controllers()
    RESULT = RF.pp_opf.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT)

def test_initialise_pandapower():
    RF = RecursiveFunction()
    RF._parameters = load_test_case()
    RF._initialise_pandapower()
    assert len(RF.pp_opf.wrapper.network['bus'].index) == 9
    assert len(RF.pp_opf.wrapper.network['controller'].iat[0, 0].data_source.df.index) == 3
    assert len(RF.pp_opf.wrapper.network['output_writer'].iat[0, 0].log_variables) == 2
    assert len(RF.pp_opf.simulation_settings.time_steps) == 3
    assert RF.pp_opf.simulation_settings.opf_type == "ac"
    assert RF.pp_opf.simulation_settings.optimisation_software == "pypower"

def test_initialise_case_pandapower():
    RF = RecursiveFunction()
    RF._initialise_pandapower = MagicMock()
    RF._create_control_graph = MagicMock()
    RF.initialise(parameters=load_test_case())
    RF._initialise_pandapower.assert_called()
    RF._create_control_graph.assert_called()

def test_initialise_case_unknown_opf():
    RF = RecursiveFunction()
    parameters=load_test_case()
    parameters.problem_settings.opf_optimizer = "other"
    RF._initialise_pandapower = MagicMock()
    RF._create_control_graph = MagicMock()
    RF.initialise(parameters=parameters)
    RF._initialise_pandapower.assert_not_called()
    RF._create_control_graph.assert_called()

def test_number_calls_methods_in_solve():
    RF = RecursiveFunction()
    parameters=load_test_case()
    parameters.optimisation_profiles_data.data.pop(-1)
    RF.initialise(parameters=parameters)
    RF._update_pandapower_controllers = MagicMock()
    RF._operational_check = MagicMock()
    RF.pp_opf.is_feasible = MagicMock()
    RF.solve(InterIterationInformation())
    assert RF._update_pandapower_controllers.call_count == 16
    assert RF._operational_check.call_count == 16

def test_create_pool_interventions(self):
    pass


def test_abstract_data_container_getitem_dict():
    data = AbstractDataContainerAppend()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    assert data["2"] == 2

def test_abstract_data_container_iterator_dict():
    data = AbstractDataContainerAppend()
    data.create_dictionary()
    data.append("1", 1)
    data.append("2", 2)
    result = []
    EXPECTED_RESULT = [["1", 1] , ["2", 2]]
    for key, value in data:
        result.append([key, value])
    assert result == EXPECTED_RESULT

def test_abstract_data_container_iterator_list():
    data = AbstractDataContainerAppend()
    data.create_list()
    data.append("1", 1)
    data.append("2", 2)
    result = []
    EXPECTED_RESULT = [["1", 1] , ["2", 2]]
    for key, value in data:
        result.append([key, value])
    assert result == EXPECTED_RESULT