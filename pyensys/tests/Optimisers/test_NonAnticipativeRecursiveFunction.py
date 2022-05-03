from pyensys.Optimisers.NonAnticipativeRecursiveFunction import NonAnticipativeRecursiveFunction
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, BinaryVariable
from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.Optimisers.NonAnticipativeRecursiveFunction import _eliminate_offsprings_of_candidate_in_incumbent, \
    _find_keys_of_offsprings_in_incumbent, _delete_offsprings_from_incumbent, _renumber_keys_in_incumbent, \
    _add_new_interventions_from_combinations, update_remaining_construction_time

from unittest.mock import MagicMock, patch
from copy import deepcopy
import pytest


def test_get_available_interventions_for_current_year():
    info = _create_dummy_information_to_test_get_available_interventions_for_current_year()
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._pool_interventions = \
        _create_dummy_pool_of_interventions_to_test_get_available_interventions_for_current_year()
    expected = AbstractDataContainer()
    expected.create_list()
    expected.append("4", BinaryVariable(installation_time=0))
    assert non_anticipative._get_available_interventions_for_current_year(info) == expected


def _create_dummy_information_to_test_get_available_interventions_for_current_year() -> \
        InterIterationInformation:
    info = InterIterationInformation()
    variables = AbstractDataContainer()
    variables.create_list()
    variables.append("0", 0)
    variables.append("1", 1)
    info.candidate_interventions.create_list()
    info.candidate_interventions.append("0", variables)
    info.candidate_interventions.append("1", AbstractDataContainer())
    info.candidate_interventions["1"].create_list()
    info.new_interventions.create_list()
    info.new_interventions.append("2", 2)
    return info


def _create_dummy_pool_of_interventions_to_test_get_available_interventions_for_current_year() -> \
        AbstractDataContainer:
    pool_interventions = AbstractDataContainer()
    pool_interventions.create_list()
    pool_interventions.append("0", 0)
    pool_interventions.append("1", 1)
    pool_interventions.append("2", 2)
    pool_interventions.append("3", BinaryVariable(installation_time=2))
    pool_interventions.append("4", BinaryVariable(installation_time=0))
    pool_interventions.append("5", BinaryVariable(installation_time=1))
    return pool_interventions


def test_verify_feasibility_of_successor_with_all_available_interventions_for_current_year():
    info = _create_dummy_information_to_test_get_available_interventions_for_current_year()
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._pool_interventions = \
        _create_dummy_pool_of_interventions_to_test_get_available_interventions_for_current_year()
    expected = AbstractDataContainer()
    expected.create_list()
    expected.append("2", 2)
    non_anticipative._operational_check = MagicMock()
    non_anticipative._is_opf_feasible = MagicMock(return_value=True)
    assert non_anticipative._verify_feasibility_of_successor_with_all_available_interventions_for_current_year(info)
    assert info.new_interventions == expected
    non_anticipative._operational_check.assert_called_once()


def test_verify_feasible_solution_in_successor_nodes_with_new_interventions():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._control_graph.graph.add_edge(100, 2)
    non_anticipative._control_graph.graph.add_edge(100, 15)
    non_anticipative._control_graph.graph.add_edge(100, 200)
    info = InterIterationInformation()
    info.level_in_graph = 0
    info.current_graph_node = 100
    non_anticipative._verify_feasibility_of_successor_with_all_available_interventions_for_current_year = MagicMock(
        return_value=True)
    assert non_anticipative._verify_feasibility_of_solution_in_successor_nodes(info)
    assert \
        non_anticipative._verify_feasibility_of_successor_with_all_available_interventions_for_current_year.call_count \
        == 3
    assert info.level_in_graph == 0
    assert info.current_graph_node == 100


def test_verify_unfeasible_solution_in_successor_nodes_with_new_interventions():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._control_graph.graph.add_edge(100, 2)
    non_anticipative._control_graph.graph.add_edge(100, 15)
    non_anticipative._control_graph.graph.add_edge(100, 200)
    info = InterIterationInformation()
    info.level_in_graph = 0
    info.current_graph_node = 100
    non_anticipative._verify_feasibility_of_successor_with_all_available_interventions_for_current_year = MagicMock(
        return_value=False)
    assert not non_anticipative._verify_feasibility_of_solution_in_successor_nodes(info)
    assert info.level_in_graph == 0
    assert info.current_graph_node == 100


@patch("pyensys.Optimisers.NonAnticipativeRecursiveFunction._add_new_interventions_from_combinations")
@pytest.mark.parametrize("feasibility_flag, expected_values", [(True, [3, 3]), (False, [3, 0])])
def test_optimise_interventions_in_last_node(mock_method, feasibility_flag, expected_values):
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._get_available_interventions_for_current_year = MagicMock(return_value=[1, 2])
    non_anticipative._check_feasibility_of_current_solution = MagicMock(return_value=feasibility_flag)
    non_anticipative._optimality_check = MagicMock()
    non_anticipative._optimise_interventions_in_last_node(InterIterationInformation())
    non_anticipative._get_available_interventions_for_current_year.assert_called_once()
    assert mock_method.call_count == 3
    assert non_anticipative._check_feasibility_of_current_solution.call_count == expected_values[0]
    assert non_anticipative._optimality_check.call_count == expected_values[1]


@pytest.mark.parametrize("feasibility_offsprings, expected_calls, return_flag, graph_feasibility, feasibility_current",
                         [(None, [0, 0], False, None, False), (False, [0, 0], False, None, True),
                          (True, [1, 1], True, True, True), (True, [1, 1], False, False, True)])
def test_exploration_of_current_solution(feasibility_offsprings, expected_calls, return_flag, graph_feasibility,
                                         feasibility_current):
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._verify_feasibility_of_solution_in_successor_nodes = MagicMock(return_value=feasibility_offsprings)
    non_anticipative._check_feasibility_of_current_solution = MagicMock(return_value=feasibility_current)
    non_anticipative._construction_of_solution = MagicMock(return_value=InterIterationInformation())
    non_anticipative._graph_exploration = MagicMock(return_value=graph_feasibility)
    assert non_anticipative._exploration_of_current_solution(InterIterationInformation()) == return_flag
    assert non_anticipative._construction_of_solution.call_count == expected_calls[0]
    assert non_anticipative._graph_exploration.call_count == expected_calls[1]


@patch("pyensys.Optimisers.NonAnticipativeRecursiveFunction._add_new_interventions_from_combinations")
@pytest.mark.parametrize("feasibility_flag, return_flag", [(True, True), (False, False)])
def test_interventions_handler(mock_method, feasibility_flag, return_flag):
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._calculate_available_interventions = MagicMock(return_value=[0, 1])
    non_anticipative._exploration_of_current_solution = MagicMock(return_value=feasibility_flag)
    assert non_anticipative._interventions_handler(InterIterationInformation()) == return_flag
    assert mock_method.call_count == 3
    assert non_anticipative._exploration_of_current_solution.call_count == 3


def _input_data_test_eliminate_offsprings_of_candidate_in_incumbent() -> InterIterationInformation:
    info = InterIterationInformation()
    info.incumbent_graph_paths.create_list()
    path = AbstractDataContainer()
    path.create_list()
    path.append("0", 0)
    path.append("1", 1)
    path.append("2", 2)
    info.incumbent_graph_paths.append("0", deepcopy(path))
    path.pop("2")
    path.append("3", 3)
    info.incumbent_graph_paths.append("1", deepcopy(path))
    path.pop("3")
    path.pop("1")
    path.append("4", 4)
    path.append("5", 5)
    info.incumbent_graph_paths.append("2", deepcopy(path))
    path.pop("4")
    path.pop("5")
    path.append("1", 1)
    info.candidate_solution_path = path
    info.incumbent_interventions.create_list()
    info.incumbent_interventions.append("0", 0)
    info.incumbent_interventions.append("1", 1)
    info.incumbent_interventions.append("2", 2)
    info.incumbent_operation_costs.create_list()
    info.incumbent_operation_costs.append("0", 0)
    info.incumbent_operation_costs.append("1", 1)
    info.incumbent_operation_costs.append("2", 2)
    info.incumbent_investment_costs.create_list()
    info.incumbent_investment_costs.append("0", 0)
    info.incumbent_investment_costs.append("1", 1)
    info.incumbent_investment_costs.append("2", 2)
    return info


def _expected_output_from_eliminate_offsprings_of_candidate_in_incumbent() -> InterIterationInformation:
    info = InterIterationInformation()
    info.incumbent_graph_paths.create_list()
    path = AbstractDataContainer()
    path.create_list()
    path.append("0", 0)
    path.append("4", 4)
    path.append("5", 5)
    info.incumbent_graph_paths.append("0", deepcopy(path))
    path.pop("5")
    path.pop("4")
    path.append("1", 1)
    info.candidate_solution_path = path
    info.incumbent_interventions.create_list()
    info.incumbent_interventions.append("0", 2)
    info.incumbent_operation_costs.create_list()
    info.incumbent_operation_costs.append("0", 2)
    info.incumbent_investment_costs.create_list()
    info.incumbent_investment_costs.append("0", 2)
    return info


def test_eliminate_offsprings_of_candidate_in_incumbent():
    info = _input_data_test_eliminate_offsprings_of_candidate_in_incumbent()
    expected = _expected_output_from_eliminate_offsprings_of_candidate_in_incumbent()
    _eliminate_offsprings_of_candidate_in_incumbent(info)
    assert info == expected


def test_find_keys_of_offsprings_in_incumbent():
    info = _input_data_test_eliminate_offsprings_of_candidate_in_incumbent()
    offsprings = _find_keys_of_offsprings_in_incumbent(info)
    assert offsprings == ["0", "1"]


def test_delete_offsprings_from_incumbent():
    info = _input_data_test_eliminate_offsprings_of_candidate_in_incumbent()
    _delete_offsprings_from_incumbent(info, ["0", "1"])
    assert info.incumbent_interventions.get("0") is None and info.incumbent_interventions.get("1") is None
    assert info.incumbent_investment_costs.get("0") is None and info.incumbent_investment_costs.get("1") is None
    assert info.incumbent_graph_paths.get("0") is None and info.incumbent_graph_paths.get("1") is None
    assert info.incumbent_operation_costs.get("0") is None and info.incumbent_operation_costs.get("1") is None


def test_renumber_keys_in_incumbent():
    info = _input_data_test_eliminate_offsprings_of_candidate_in_incumbent()
    _delete_offsprings_from_incumbent(info, ["0", "1"])
    _renumber_keys_in_incumbent(info)
    assert info.incumbent_graph_paths.get("0").get("0") == 0
    assert info.incumbent_graph_paths.get("0").get("4") == 4
    assert info.incumbent_graph_paths.get("0").get("5") == 5
    assert info.incumbent_investment_costs.get("0") == 2
    assert info.incumbent_interventions.get("0") == 2
    assert info.incumbent_operation_costs.get("0") == 2


def test_add_new_interventions_from_combinations():
    interventions: AbstractDataContainer = AbstractDataContainer()
    interventions.create_list()
    interventions.append("0", BinaryVariable(installation_time=2))
    rf = NonAnticipativeRecursiveFunction()
    combination = rf._calculate_all_combinations(available_interventions=interventions, length_set=1)
    info = InterIterationInformation()
    _add_new_interventions_from_combinations(info, combination)
    assert len(info.new_interventions) == 1
    assert info.new_interventions.get("0") == BinaryVariable(installation_time=2)
    assert info.new_interventions_remaining_construction_time.get("0") == 2


@pytest.mark.parametrize("flag, expected", [(True, True), (False, False)])
def test_check_feasibility_of_current_solution(flag, expected):
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._operational_check = MagicMock()
    non_anticipative._is_opf_feasible = MagicMock(return_value=flag)
    assert non_anticipative._check_feasibility_of_current_solution(InterIterationInformation()) == expected


def test_update_remaining_construction_time():
    info = InterIterationInformation()
    info.candidate_interventions_remaining_construction_time.create_list()
    info.candidate_interventions_remaining_construction_time.append("0", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["0"].create_list()
    info.candidate_interventions_remaining_construction_time["0"].append("0", 1)
    info.candidate_interventions_remaining_construction_time["0"].append("1", 2)
    info.candidate_interventions_remaining_construction_time.append("1", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["1"].create_list()
    info.new_interventions_remaining_construction_time.create_list()
    info.new_interventions_remaining_construction_time.append("2", 3)
    info.new_interventions_remaining_construction_time.append("3", 4)
    update_remaining_construction_time(info)
    assert info.candidate_interventions_remaining_construction_time.get("0").get("0") == 0
    assert info.candidate_interventions_remaining_construction_time.get("0").get("1") == 1
    assert info.new_interventions_remaining_construction_time.get("2") == 2
    assert info.new_interventions_remaining_construction_time.get("3") == 3


def test_type_error_update_remaining_construction_time():
    with pytest.raises(TypeError):
        update_remaining_construction_time(list())


def test_get_interventions_ready_to_operate_in_opf():
    info = InterIterationInformation()
    info.candidate_interventions_remaining_construction_time.create_list()
    info.candidate_interventions_remaining_construction_time.append("0", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["0"].create_list()
    info.candidate_interventions_remaining_construction_time["0"].append("0", 0)
    info.candidate_interventions_remaining_construction_time["0"].append("1", 2)
    info.candidate_interventions_remaining_construction_time.append("1", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["1"].create_list()
    info.new_interventions_remaining_construction_time.create_list()
    info.new_interventions_remaining_construction_time.append("2", 0)
    info.new_interventions_remaining_construction_time.append("3", 3)
    info.candidate_interventions.create_list()
    info.candidate_interventions.append("0", AbstractDataContainer())
    info.candidate_interventions["0"].create_list()
    info.candidate_interventions["0"].append("0", BinaryVariable(variable_name="a"))
    info.candidate_interventions["0"].append("1", BinaryVariable(variable_name="b"))
    info.new_interventions.create_list()
    info.new_interventions.append("2", BinaryVariable(variable_name="c"))
    info.new_interventions.append("3", BinaryVariable(variable_name="d"))
    non_anticipative = NonAnticipativeRecursiveFunction()
    interventions = non_anticipative._get_interventions_ready_to_operate_in_opf(info)
    assert len(interventions) == 2
    assert  interventions[0].variable_name == "a"
    assert  interventions[1].variable_name == "c"

def test_create_list_of_parameters_to_update_in_opf():
    variables = []
    variables.append(BinaryVariable(element_type="a"))
    variables.append(BinaryVariable(element_position=0))
    non_anticipative = NonAnticipativeRecursiveFunction()
    parameters = non_anticipative._create_list_of_parameters_to_update_in_opf(variables)
    assert len(parameters) == 2
    assert parameters[0].component_type == "a"
    assert parameters[1].parameter_position == 0

def test_update_status_elements_opf():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._get_interventions_ready_to_operate_in_opf = MagicMock()
    non_anticipative._create_list_of_parameters_to_update_in_opf = MagicMock()
    non_anticipative._opf.update_multiple_parameters = MagicMock()
    dummy = InterIterationInformation()
    dummy.new_interventions.create_list()
    dummy.new_interventions.append("0", 0)
    non_anticipative._update_status_elements_opf(dummy)
    non_anticipative._get_interventions_ready_to_operate_in_opf.assert_called_once()
    non_anticipative._create_list_of_parameters_to_update_in_opf.assert_called_once()
    non_anticipative._opf.update_multiple_parameters.assert_called_once()

def test_update_remaining_construction_time_by_increasing_one_period():
    info = InterIterationInformation()
    info.candidate_interventions_remaining_construction_time.create_list()
    info.candidate_interventions_remaining_construction_time.append("0", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["0"].create_list()
    info.candidate_interventions_remaining_construction_time["0"].append("0", 1)
    info.candidate_interventions_remaining_construction_time.append("1", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["1"].create_list()
    info.new_interventions_remaining_construction_time.create_list()
    info.new_interventions_remaining_construction_time.append("2", 3)
    update_remaining_construction_time(info, 1)
    assert info.candidate_interventions_remaining_construction_time.get("0").get("0") == 2
    assert info.new_interventions_remaining_construction_time.get("2") == 4

def test_append_candidate_interventions_in_incumbent_interventions_list():
    info = InterIterationInformation()
    info.incumbent_interventions.create_list()
    info.incumbent_graph_paths.create_list()
    info.incumbent_investment_costs.create_list()
    info.incumbent_operation_costs.create_list()
    info.candidate_interventions.create_list()
    info.candidate_interventions.append("0", AbstractDataContainer())
    info.candidate_interventions["0"].create_list()
    info.candidate_interventions["0"].append("0", BinaryVariable(variable_name="c"))
    info.candidate_interventions["0"].append("1", BinaryVariable(variable_name="d"))
    info.candidate_interventions["0"].append("2", BinaryVariable(variable_name="e"))
    info.candidate_interventions_remaining_construction_time.create_list()
    info.candidate_interventions_remaining_construction_time.append("0", AbstractDataContainer())
    info.candidate_interventions_remaining_construction_time["0"].create_list()
    info.candidate_interventions_remaining_construction_time["0"].append("0", 0)
    info.candidate_interventions_remaining_construction_time["0"].append("1", 1)
    info.candidate_interventions_remaining_construction_time["0"].append("2", -1)
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._append_candidate_interventions_in_incumbent_interventions_list("0", info)
    assert len(info.incumbent_interventions) == 1
    assert len(info.incumbent_interventions["0"]["0"]) == 2
    assert info.incumbent_interventions["0"]["0"]["0"].variable_name == "c"
    assert info.incumbent_interventions["0"]["0"]["2"].variable_name == "e"
