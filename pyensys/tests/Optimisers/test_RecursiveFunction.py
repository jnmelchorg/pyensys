from numpy.testing._private.utils import assert_equal
from pyensys.Optimisers.RecursiveFunction import *
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable, \
    OptimisationProfileData, OptimisationBinaryVariables
from pyensys.tests.test_data_paths import get_path_case9_mat, set_pandapower_test_output_directory, \
    get_clustering_data_test
from pyensys.Optimisers.ControlGraphsCreator import ClusterData

from pandas import DataFrame, date_range, read_excel
from math import isclose
from typing import List, Dict
from unittest.mock import MagicMock
from pytest import raises


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
    RESULT = RF._create_new_pandapower_profiles(1)
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT.data[1].data)

def test_initialise_pandapower():
    RF = RecursiveFunction()
    RF._parameters = load_test_case()
    RF._initialise_pandapower()
    assert len(RF._opf.wrapper.network['bus'].index) == 9
    assert len(RF._opf.wrapper.network['controller'].iat[0, 0].data_source.df.index) == 3
    assert len(RF._opf.wrapper.network['output_writer'].iat[0, 0].log_variables) == 2
    assert len(RF._opf.simulation_settings.time_steps) == 3
    assert RF._opf.simulation_settings.opf_type == "ac"
    assert RF._opf.simulation_settings.optimisation_software == "pypower"

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

def _create_dummy_optimisation_binary_variables() -> RecursiveFunction:
    RF = RecursiveFunction()
    RF._parameters.initialised = True
    RF._parameters.optimisation_binary_variables = [
        OptimisationBinaryVariables(element_type="gen", variable_name="installation", 
        elements_ids=["G0", "G1"], elements_positions=[0, 1], costs=[1.0, 2.0], installation_time=\
        [0,0]),
        OptimisationBinaryVariables(element_type="AC line", variable_name="installation", 
        elements_ids=["L0", "L1"], elements_positions=[0, 1], costs=[5.0, 6.0], installation_time=\
        [0,0])
    ]
    return RF

def test_create_pool_interventions():
    RF = _create_dummy_optimisation_binary_variables()
    RF._create_pool_interventions()
    assert_equal(len(RF._pool_interventions), 4)
    assert isinstance(RF._pool_interventions["2"], BinaryVariable)
    assert_equal(RF._pool_interventions["1"].element_type, "gen")
    assert_equal(RF._pool_interventions["1"].variable_name, "installation")
    assert_equal(RF._pool_interventions["1"].element_id, "G1")
    assert_equal(RF._pool_interventions["1"].element_position, 1)
    assert_equal(RF._pool_interventions["1"].cost, 2.0)
    assert_equal(RF._pool_interventions["3"].cost, 6.0)

def test_calculate_all_combinations():
    AO = AbstractDataContainer()
    AO.create_list()
    for x in range(0,3):
        AO.append(str(x), x)
    RF = RecursiveFunction()
    assert list(RF._calculate_all_combinations(AO, 2)) == [(("0", 0),("1", 1),), (("0", 0),("2", 2),), (("1", 1),("2", 2),)]

def test_get_total_operation_cost():
    RF = RecursiveFunction()
    RF._opf = PandaPowerManager()
    RF._opf.get_total_cost = MagicMock(return_value=2.5)
    assert RF._get_total_operation_cost() == 912.5

def _create_dummy_pool_of_interventions() -> RecursiveFunction:
    RF = _create_dummy_optimisation_binary_variables()
    RF._create_pool_interventions()
    return RF

def test_calculate_investment_cost():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    investments = AbstractDataContainer()
    investments.create_list()
    investments.append("0", RF._pool_interventions["0"])
    investments.append("1", RF._pool_interventions["2"])
    interventions = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0", investments)
    investments = AbstractDataContainer()
    investments.create_list()
    investments.append("0", RF._pool_interventions["1"])
    investments.append("1", RF._pool_interventions["3"])
    interventions.append("1", investments)
    assert isclose(RF._calculate_investment_cost(interventions), 14.247, rel_tol=1e-3)

def test_calculate_opteration_cost():
    RF = RecursiveFunction()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    operation = AbstractDataContainer()
    operation.create_list()
    operation.append("0", 200.0)
    operation.append("1", 300.0)
    assert isclose(RF._calculate_opteration_cost(operation), 509.278, rel_tol=1e-3)

def _create_dummy_candidate_solution(RF: RecursiveFunction) -> InterIterationInformation:
    inter_information = InterIterationInformation()
    inter_information.candidate_interventions.create_list()
    investments = AbstractDataContainer()
    investments.create_list()
    investments.append("0", RF._pool_interventions["0"])
    investments.append("1", RF._pool_interventions["2"])
    inter_information.candidate_interventions.append("0", investments)
    investments = AbstractDataContainer()
    investments.create_list()
    investments.append("0", RF._pool_interventions["1"])
    investments.append("1", RF._pool_interventions["3"])
    inter_information.candidate_interventions.append("1", investments)
    inter_information.candidate_operation_cost.create_list()
    inter_information.candidate_operation_cost.append("0", 200)
    inter_information.candidate_operation_cost.append("1", 300)
    inter_information.candidate_solution_path.create_list()
    inter_information.candidate_solution_path.append("0", 0)
    inter_information.candidate_solution_path.append("1", 1)
    return inter_information

def _create_empty_incumbent_list(inter_information: InterIterationInformation) -> \
    InterIterationInformation:
    inter_information.incumbent_interventions.create_list()
    inter_information.incumbent_graph_paths.create_list()
    inter_information.incumbent_investment_costs.create_list()
    inter_information.incumbent_operation_costs.create_list()
    return inter_information

def test_optimality_check_case_empty_incumbent():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_empty_incumbent_list(inter_information)    
    RF._optimality_check(inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 509.278, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 14.247, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.incumbent_interventions["0"]) == 2
    assert len(inter_information.candidate_interventions) == 1
    assert len(inter_information.candidate_solution_path) == 1
    assert len(inter_information.candidate_operation_cost) == 1

def _create_incumbent_list_with_one_better_solution(inter_information: InterIterationInformation) -> \
    InterIterationInformation:
    inter_information.incumbent_interventions.create_list()
    inter_information.incumbent_interventions = deepcopy(inter_information.candidate_interventions)
    inter_information.incumbent_graph_paths.create_list()
    inter_information.incumbent_graph_paths.append("0", AbstractDataContainer())
    inter_information.incumbent_graph_paths["0"].create_list()
    inter_information.incumbent_graph_paths["0"].append("0", 0)
    inter_information.incumbent_graph_paths["0"].append("1", 1)
    inter_information.incumbent_investment_costs.create_list()
    inter_information.incumbent_investment_costs.append("0", 10)
    inter_information.incumbent_operation_costs.create_list()
    inter_information.incumbent_operation_costs.append("0", 500)
    return inter_information

def test_optimality_check_case_existing_solution_in_incumbent_candidate_is_worse():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_incumbent_list_with_one_better_solution(inter_information)    
    RF._optimality_check(inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 500, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 10, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.candidate_interventions) == 1
    assert len(inter_information.candidate_solution_path) == 1
    assert len(inter_information.candidate_operation_cost) == 1

def _create_incumbent_list_with_one_worse_solution(inter_information: InterIterationInformation) -> \
    InterIterationInformation:
    inter_information = _create_incumbent_list_with_one_better_solution(inter_information)
    inter_information.incumbent_investment_costs["0"] = 20
    inter_information.incumbent_operation_costs["0"] = 520
    return inter_information

def test_optimality_check_case_existing_solution_in_incumbent_candidate_is_better():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_incumbent_list_with_one_worse_solution(inter_information)
    RF._optimality_check(inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 509.278, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 14.247, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.incumbent_interventions["0"]) == 2
    assert len(inter_information.candidate_interventions) == 1
    assert len(inter_information.candidate_solution_path) == 1
    assert len(inter_information.candidate_operation_cost) == 1

def test_append_candidate_in_incumbent_list():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_empty_incumbent_list(inter_information) 
    RF._append_candidate_in_incumbent_list("0", inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 509.278, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 14.247, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.incumbent_interventions["0"]) == 2

def test_check_if_candidate_path_has_been_stored_in_incumbent():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_incumbent_list_with_one_worse_solution(inter_information)
    key = RF._check_if_candidate_path_has_been_stored_in_incumbent(inter_information)
    assert key == "0"

def test_replace_incumbent_if_candidate_is_better():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_incumbent_list_with_one_worse_solution(inter_information)
    RF._replace_incumbent_if_candidate_is_better("0", inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 509.278, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 14.247, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.incumbent_interventions["0"]) == 2

def test_replace_solution_in_incumbent_list():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information = _create_incumbent_list_with_one_worse_solution(inter_information)
    RF._replace_solution_in_incumbent_list("0", inter_information)
    assert isclose(inter_information.incumbent_operation_costs["0"], 509.278, rel_tol=1e-3)
    assert isclose(inter_information.incumbent_investment_costs["0"], 14.247, rel_tol=1e-3)
    assert inter_information.incumbent_graph_paths["0"]["0"] == 0
    assert inter_information.incumbent_graph_paths["0"]["1"] == 1
    assert len(inter_information.incumbent_interventions["0"]) == 2

def test_return_to_previous_state():
    RF = _create_dummy_pool_of_interventions()
    RF._parameters.problem_settings.return_rate_in_percentage = 3.0
    inter_information = _create_dummy_candidate_solution(RF)
    inter_information.level_in_graph = 1
    RF._return_to_previous_state(inter_information)
    assert len(inter_information.candidate_interventions) == 1
    assert len(inter_information.candidate_solution_path) == 1
    assert len(inter_information.candidate_operation_cost) == 1

def test_add_new_interventions_from_combinations():
    combination:List[AbstractDataContainer] = []
    combination.append(AbstractDataContainer())
    combination[0].create_list()
    combination[0].append("0", 0)
    info = InterIterationInformation()
    RF = RecursiveFunction()
    RF._add_new_interventions_from_combinations(info, combination)
    assert_equal(len(info.new_interventions), 1)
    assert_equal(info.new_interventions.get("0"), 0)

def test_graph_exploration():
    RF = RecursiveFunction()
    RF._control_graph.graph.add_edge(100,5)
    RF._control_graph.graph.add_edge(100,20)
    info = InterIterationInformation()
    info.current_graph_node = 100
    info.level_in_graph = 2
    RF.solve = MagicMock()
    RF._graph_exploration(info)
    assert_equal(info.level_in_graph, 4)
    assert_equal(RF.solve.call_count, 2)
    assert_equal(info.current_graph_node, 100)

def test_calculate_available_interventions():
    info = _create_dummy_inter_iteration_information_for_test_calculate_available_interventions()
    recursive_f = RecursiveFunction()
    recursive_f._pool_interventions = _create_dummy_pool_of_interventions_for_test_calculate_available_interventions()
    expected = AbstractDataContainer()
    expected.create_list()
    expected.append("1", 1)
    assert_equal(recursive_f._calculate_available_interventions(info), expected)

def _create_dummy_inter_iteration_information_for_test_calculate_available_interventions() -> InterIterationInformation:
    info = InterIterationInformation()
    info.candidate_interventions.create_list()
    info.candidate_interventions.append("0", AbstractDataContainer())
    info.candidate_interventions["0"].create_list()
    info.candidate_interventions["0"].append("2", 2)
    empty = AbstractDataContainer()
    empty.create_list()
    info.candidate_interventions.append("1", empty)
    info.new_interventions.create_list()
    info.new_interventions.append("0", 0)
    return info

def _create_dummy_pool_of_interventions_for_test_calculate_available_interventions() -> \
    AbstractDataContainer:
    interventions = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0",0)
    interventions.append("1",1)
    interventions.append("2",2)
    return interventions

def test_interventions_handler():
    recursive_f = RecursiveFunction()
    recursive_f._calculate_available_interventions = MagicMock(return_value=[0, 1])
    recursive_f._graph_exploration = MagicMock()
    recursive_f._add_new_interventions_from_combinations = MagicMock()
    recursive_f._interventions_handler(InterIterationInformation())
    recursive_f._calculate_available_interventions.assert_called_once()
    assert_equal(recursive_f._graph_exploration.call_count, 4)
    assert_equal(recursive_f._add_new_interventions_from_combinations.call_count, 3)

def test_construction_of_solution():
    recursive_f = RecursiveFunction()
    recursive_f._get_total_operation_cost = MagicMock(return_value=10)
    info = _dummy_inter_iteration_information_for_test_construction_of_solution()
    info = recursive_f._construction_of_solution(info)
    assert_equal(len(info.candidate_solution_path), 3)
    assert_equal(info.candidate_solution_path.get("2"), 4)
    assert_equal(info.candidate_interventions["2"]["3"], 3)
    assert_equal(info.candidate_operation_cost["2"], 10)

def _dummy_inter_iteration_information_for_test_construction_of_solution() -> InterIterationInformation:
    info = InterIterationInformation()
    info.level_in_graph = 2
    info.current_graph_node = 4
    info.candidate_solution_path.create_list()
    info.candidate_solution_path.append("0", 0)
    info.candidate_solution_path.append("1", 1)
    info.new_interventions.create_list()
    info.new_interventions.append("3", 3)
    info.candidate_interventions.create_list()
    info.candidate_operation_cost.create_list()
    return info

def test_is_opf_feasible():
    recursive_f = RecursiveFunction()
    recursive_f._opf.is_feasible = MagicMock(return_value=True)
    assert recursive_f._is_opf_feasible()

def test_operational_check():
    recursive = RecursiveFunction()
    info = InterIterationInformation()
    recursive._update_status_elements_opf = MagicMock()
    recursive._update_pandapower_controllers = MagicMock()
    recursive._run_opf = MagicMock()
    recursive._operational_check(info)
    recursive._update_status_elements_opf.assert_called_once()
    recursive._update_pandapower_controllers.assert_called_once()
    recursive._run_opf.assert_called_once()

def test_update_status_elements_opf_with_new_interventions():
    recursive = RecursiveFunction()
    info = InterIterationInformation()
    info.new_interventions.create_list()
    info.new_interventions.append("0", 0)
    recursive._update_status_elements_opf_per_intervention_group = MagicMock()
    recursive._update_status_elements_opf(info)
    assert_equal(recursive._update_status_elements_opf_per_intervention_group.call_count, 2)

def test_update_status_elements_opf_without_new_interventions():
    recursive = RecursiveFunction()
    info = InterIterationInformation()
    info.new_interventions.create_list()
    recursive._update_status_elements_opf_per_intervention_group = MagicMock()
    recursive._update_status_elements_opf(info)
    recursive._update_status_elements_opf_per_intervention_group.assert_not_called()

def test_update_status_elements_opf_per_intervention_group_with_binary_variablest():
    recursive = RecursiveFunction()
    interventions = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0", BinaryVariable())
    interventions.append("1", BinaryVariable())
    recursive._opf.update_multiple_parameters = MagicMock()
    recursive._create_data_to_update_parameter = MagicMock(return_value=0)
    recursive._update_status_elements_opf_per_intervention_group(interventions)
    assert_equal(recursive._create_data_to_update_parameter.call_count, 2)
    recursive._opf.update_multiple_parameters.assert_called_once()

def test_update_status_elements_opf_per_intervention_group_with_abstract_data_object_of_binary_variables():
    recursive = RecursiveFunction()
    interventions = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0", BinaryVariable())
    interventions.append("1", BinaryVariable())
    path_investments = AbstractDataContainer()
    path_investments.create_list()
    path_investments.append("0", interventions)
    path_investments.append("1", interventions)
    recursive._opf.update_multiple_parameters = MagicMock()
    recursive._create_data_to_update_parameter = MagicMock(return_value=0)
    recursive._update_status_elements_opf_per_intervention_group(path_investments)
    assert_equal(recursive._create_data_to_update_parameter.call_count, 4)
    recursive._opf.update_multiple_parameters.assert_called_once()

def test_type_error_update_status_elements_opf_per_intervention_group():
    recursive = RecursiveFunction()
    interventions = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0", [])
    recursive._opf.update_multiple_parameters = MagicMock()
    with raises(TypeError):
        recursive._update_status_elements_opf_per_intervention_group(interventions)
    recursive._opf.update_multiple_parameters.assert_not_called()

def test_create_data_to_update_parameter():
    recursive = RecursiveFunction()
    var = _create_dummy_binary_variable_to_test_create_data_to_update_parameter()
    param = recursive._create_data_to_update_parameter(var)
    assert_equal(param.component_type, "dem")
    assert_equal(param.parameter_position, 2)

def _create_dummy_binary_variable_to_test_create_data_to_update_parameter() -> BinaryVariable:
    var = BinaryVariable()
    var.element_type = "dem"
    var.element_position = 2
    return var

def test_type_error_in_create_data_to_update_parameter_with_wrong_input():
    recursive = RecursiveFunction()
    with raises(TypeError):
        recursive._create_data_to_update_parameter(list())

def test_update_pandapower_controllers():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._control_graph.nodes_data = load_data_multipliers()
    RF._update_pandapower_controllers(1)
    RESULT = RF._opf.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT)

def test_run_opf():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF._run_opf()
    assert RF._opf.wrapper.network.OPF_converged == True
    assert isclose(3583.53647, RF._opf.wrapper.network.res_cost, abs_tol=1e-4)
