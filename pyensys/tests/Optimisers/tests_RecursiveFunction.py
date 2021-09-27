from pyensys.Optimisers.RecursiveFunction import *
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable, \
    OptimisationProfileData
from pyensys.tests.tests_data_paths import get_path_case9_mat, set_pandapower_test_output_directory, \
    get_clustering_data_test
from pandas import DataFrame, date_range, read_excel, option_context
from math import isclose
from typing import List
from copy import copy
from numpy import allclose

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

def load_data_multipliers() -> List[DataMultipliers]:
    list_multipliers = []
    data_multipliers = DataMultipliers()
    data_multipliers.initialised = True
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.1, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.2, element_type="gen", variable_name="p_mw")]
    list_multipliers.append(copy(data_multipliers))
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.3, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.4, element_type="gen", variable_name="p_mw")]
    list_multipliers.append(data_multipliers)
    return list_multipliers

def test_initialise():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    assert len(RF.pp_opf.wrapper.network['bus'].index) == 9
    assert len(RF.pp_opf.wrapper.network['controller'].iat[0, 0].data_source.df.index) == 3
    assert len(RF.pp_opf.wrapper.network['output_writer'].iat[0, 0].log_variables) == 2
    assert len(RF.pp_opf.simulation_settings.time_steps) == 3
    assert RF.pp_opf.simulation_settings.opf_type == "ac"
    assert RF.pp_opf.simulation_settings.optimisation_software == "pypower"

def test_operational_check():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF.operational_check()
    assert RF.pp_opf.wrapper.network.OPF_converged == True
    assert isclose(3583.53647, RF.pp_opf.wrapper.network.res_cost, abs_tol=1e-4)

def test_get_data_multipliers_current_node():
    RF = RecursiveFunction()
    data_multipliers = DataMultipliers()
    data_multipliers.initialised = True
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.1, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.2, element_type="load", variable_name="p_mw")]
    RF.graphs_multipliers.append(data_multipliers)
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.3, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.4, element_type="load", variable_name="p_mw")]
    RF.graphs_multipliers.append(data_multipliers)
    assert RF.get_data_multipliers_current_node(1).multipliers[0].data_multiplier == 0.3

def test_get_profile_position_to_update():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    list_multipliers = load_data_multipliers()
    PROFILE_POSITION = RF.get_profile_position_to_update(list_multipliers[1].multipliers[1])
    assert PROFILE_POSITION == 1

def test_create_profile_clusters():
    parameters = load_test_case()
    profile_clusters = create_profile_clusters(parameters.optimisation_profiles_data.data[0])
    with option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(profile_clusters)
        print(profile_clusters.nunique(axis=1).tolist())
    EXPECTED_NUMBER_UNIQUE_VALUES = [1, 1, 1, 2, 3, 2, 3]
    EXPECTED_VALUES_ROW_4 = [0.85, 1.04, 1.4]
    assert profile_clusters.shape == (7, 3)
    assert profile_clusters.nunique(axis=1).tolist() == EXPECTED_NUMBER_UNIQUE_VALUES
    assert allclose(profile_clusters.loc[4].tolist(), EXPECTED_VALUES_ROW_4)

def test_create_profiles_clusters():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    assert len(profiles_clusters.data) == 2
    assert (profiles_clusters.data[1].variable_name == "p_mw")
    assert (profiles_clusters.data[1].element_type == "gen")

def test_update_pandapower_controllers():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF.graphs_multipliers = load_data_multipliers()
    RF.update_pandapower_controllers(1)
    RESULT = RF.pp_opf.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT)

def test_determine_graph_nodes_in_row():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = determine_graph_nodes_in_row(profiles_clusters.data[0].data.loc[3])
    assert nodes[0].scenarios == [0] and nodes[1].scenarios == [1, 2]
    assert nodes[0].multiplier == 0.9 and nodes[1].multiplier == 1.025

def test_determine_graph_nodes():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    graph_nodes = determine_graph_nodes(profiles_clusters.data[0].data)
    assert graph_nodes.initialised
    assert len(graph_nodes.nodes) == 13
    assert graph_nodes.nodes[0].scenarios == [0, 1, 2]
    assert graph_nodes.nodes[3].scenarios == [0] and graph_nodes.nodes[4].scenarios == [1, 2]
    assert graph_nodes.nodes[8].multiplier == 0.98
    assert graph_nodes.nodes[6].level == 4

def test_get_nodes_per_level():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    graph_nodes = determine_graph_nodes(profiles_clusters.data[0].data)
    nodes_per_level = get_nodes_per_level(graph_nodes)
    assert len(nodes_per_level) == 7
    assert nodes_per_level[4] == [5, 6, 7] and nodes_per_level[6] == [10, 11, 12]

def test_determine_graph_edges():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    graph_nodes = determine_graph_nodes(profiles_clusters.data[0].data)
    edges = determine_graph_edges(graph_nodes)
    assert len(edges) == 13
    assert edges[2] == [2, 3] and edges[6] == [4, 7] and edges[11] == [8, 11]

def test_get_graph_multipliers():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    graph_nodes = determine_graph_nodes(profiles_clusters.data[0].data)
    multipliers = get_graph_multipliers(graph_nodes, profiles_clusters.data[0])
    assert len(multipliers.multipliers) == 13
    assert multipliers.multipliers[0].data_multiplier == 1.0 and \
        multipliers.multipliers[6].data_multiplier == 1.04

def test_create_graph():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    RF = RecursiveFunction()
    RF.create_graph(profiles_clusters.data[0])
    assert RF.graphs[0].number_of_nodes() == 13
    assert len(RF.graphs_multipliers[0].multipliers) == 13
