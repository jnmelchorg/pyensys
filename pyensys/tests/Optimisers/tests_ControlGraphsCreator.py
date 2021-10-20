from pyensys.Optimisers.ControlGraphsCreator import *
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable, \
    OptimisationProfileData
from pyensys.tests.tests_data_paths import get_path_case9_mat, set_pandapower_test_output_directory, \
    get_clustering_data_test

from pandas import DataFrame, date_range, read_excel
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

def test_determine_nodes_in_profile():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = GraphNodesCreator()
    nodes._profile=profiles_clusters.data[0]
    nodes._determine_nodes_in_profile()
    assert len(nodes._nodes_in_dataframe) == 7
    assert len(nodes._nodes_in_dataframe[3]) == 2
    assert nodes._nodes_in_dataframe[3][0].scenarios == [0] and \
        nodes._nodes_in_dataframe[3][1].scenarios == [1, 2]
    assert nodes._nodes_in_dataframe[3][0].centroid == 0.9 and \
        nodes._nodes_in_dataframe[3][1].centroid == 1.025

def test_add_final_information_to_nodes():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = GraphNodesCreator()
    nodes._profile=profiles_clusters.data[0]
    nodes._determine_nodes_in_profile()
    nodes._add_final_information_to_nodes()
    assert len(nodes._graph_nodes) == 13
    assert nodes._graph_nodes[0].scenarios == [0, 1, 2]
    assert nodes._graph_nodes[3].scenarios == [0] and nodes._graph_nodes[4].scenarios == [1, 2]
    assert nodes._graph_nodes[8].centroid == 0.98
    assert nodes._graph_nodes[6].level == 4
    assert nodes._graph_nodes[6].element_type == "load"
    assert nodes._graph_nodes[6].variable_name == "p_mw"

def test_determine_graph_nodes():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = GraphNodesCreator()
    graph_nodes = nodes.determine_graph_nodes(profiles_clusters.data[0])
    assert len(graph_nodes) == 13
    assert graph_nodes[0].scenarios == [0, 1, 2]
    assert graph_nodes[3].scenarios == [0] and graph_nodes[4].scenarios == [1, 2]
    assert graph_nodes[8].centroid == 0.98
    assert graph_nodes[6].level == 4
    assert graph_nodes[6].element_type == "load"
    assert graph_nodes[6].variable_name == "p_mw"

def test_get_nodes_per_level():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = GraphNodesCreator()
    graph_nodes = nodes.determine_graph_nodes(profiles_clusters.data[0])
    edges = GraphEdgesCreator()
    edges._graph_nodes = graph_nodes
    edges._get_nodes_all_levels()
    assert len(edges._nodes_all_levels) == 7
    assert edges._nodes_all_levels[4] == [5, 6, 7] and edges._nodes_all_levels[6] == [10, 11, 12]

def test_determine_graph_edges():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    nodes = GraphNodesCreator()
    graph_nodes = nodes.determine_graph_nodes(profiles_clusters.data[0])
    edges = GraphEdgesCreator()
    graph_edges = edges.determine_graph_edges(graph_nodes)
    assert len(graph_edges) == 13
    assert graph_edges[2].end1 == 2 and graph_edges[2].end2 == 3
    assert graph_edges[6].end1 == 4 and graph_edges[6].end2 == 7
    assert graph_edges[11].end1 == 8 and graph_edges[11].end2 == 11

def test_create_nodes_and_edges():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    profile_graph = ProfileGraphCreator()
    profile_graph._profile = profiles_clusters.data[0]
    profile_graph._create_nodes_and_edges()
    assert len(profile_graph._edges) == 13
    assert profile_graph._edges[2].end1 == 2 and profile_graph._edges[2].end2 == 3
    assert len(profile_graph._profile_graph.nodes_data) == 13
    assert profile_graph._profile_graph.nodes_data[3].scenarios == [0] and \
        profile_graph._profile_graph.nodes_data[4].scenarios == [1, 2]
    assert profile_graph._profile_graph.nodes_data[8].centroid == 0.98
    assert profile_graph._profile_graph.nodes_data[6].level == 4
    assert profile_graph._profile_graph.nodes_data[6].element_type == "load"
    assert profile_graph._profile_graph.nodes_data[6].variable_name == "p_mw"

def test_create_graph():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    profile_graph = ProfileGraphCreator()
    graph_data = profile_graph.create_graph(profiles_clusters.data[0])
    assert graph_data.graph.number_of_nodes() == 13
    assert graph_data.nodes_data[3].scenarios == [0] and \
        graph_data.nodes_data[4].scenarios == [1, 2]
    assert graph_data.nodes_data[8].centroid == 0.98

def test_create_all_profile_graphs():
    parameters = load_test_case()
    profiles_clusters = create_profiles_clusters(parameters)
    all_profiles_graphs_creator = AllProfilesGraphsCreator()
    profiles_graphs = all_profiles_graphs_creator.create_all_profile_graphs(profiles_clusters)
    assert len(profiles_graphs) == 2
    assert profiles_graphs[0].graph.number_of_nodes() == 13
    assert profiles_graphs[1].graph.number_of_nodes() == 13

def test_create_profile_clusters():
    parameters = load_test_case()
    profile_clusters = create_profile_clusters(parameters.optimisation_profiles_data.data[0])
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

def test_create_all_profiles_graphs():
    parameters = load_test_case()
    control_graph = RecursiveFunctionGraphCreator()
    control_graph._profiles = create_profiles_clusters(parameters)
    control_graph._create_all_profiles_graphs()
    assert len(control_graph._profiles_graphs) == 2
    assert control_graph._profiles_graphs[0].graph.number_of_nodes() == 13
    assert control_graph._profiles_graphs[1].graph.number_of_nodes() == 13

def test_create_control_graph():
    parameters = load_test_case()
    control_graph = RecursiveFunctionGraphCreator()
    parameters.optimisation_profiles_data.data.pop()
    control_graph._profiles = create_profiles_clusters(parameters)
    control_graph._create_all_profiles_graphs()
    control_graph._create_control_graph()
    assert control_graph._control_graph.graph.number_of_nodes() == 13
    assert control_graph._control_graph.nodes_data[3][0].scenarios == [0] and \
        control_graph._control_graph.nodes_data[4][0].scenarios == [1, 2]
    assert control_graph._control_graph.nodes_data[8][0].centroid == 0.98

def test_create_recursive_function_graph():
    parameters = load_test_case()
    parameters.optimisation_profiles_data.data.pop()
    control_graph = RecursiveFunctionGraphCreator()
    recursive_function_graph = control_graph.create_recursive_function_graph(parameters)
    assert recursive_function_graph.graph.number_of_nodes() == 13
    assert recursive_function_graph.nodes_data[3][0].scenarios == [0] and \
        recursive_function_graph.nodes_data[4][0].scenarios == [1, 2]
    assert control_graph._control_graph.nodes_data[8][0].centroid == 0.98