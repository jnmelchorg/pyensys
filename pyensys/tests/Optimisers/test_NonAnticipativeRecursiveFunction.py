from multiprocessing import pool
from pyensys.Optimisers.NonAnticipativeRecursiveFunction import NonAnticipativeRecursiveFunction
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, BinaryVariable
from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer

from unittest.mock import MagicMock

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

def test_verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year():
    info = _create_dummy_information_to_test_get_available_interventions_for_current_year()
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._pool_interventions = \
        _create_dummy_pool_of_interventions_to_test_get_available_interventions_for_current_year()
    expected = AbstractDataContainer()
    expected.create_list()
    expected.append("2", 2)
    non_anticipative._operational_check = MagicMock()
    non_anticipative._is_opf_feasible = MagicMock(return_value=True)
    assert non_anticipative._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year(info)
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
    non_anticipative._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year = MagicMock(return_value=True)
    assert non_anticipative._verify_feasibility_of_solution_in_successor_nodes(info)
    assert non_anticipative._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year.call_count == 3
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
    non_anticipative._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year = MagicMock(return_value=False)
    assert not non_anticipative._verify_feasibility_of_solution_in_successor_nodes(info)
    assert info.level_in_graph == 0
    assert info.current_graph_node == 100
    
def test_feasible_solution_in_check_optimality_and_feasibility_of_current_solution():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._operational_check = MagicMock()
    non_anticipative._is_opf_feasible = MagicMock(return_value=True)
    non_anticipative._optimality_check = MagicMock()
    non_anticipative._check_optimality_and_feasibility_of_current_solution(InterIterationInformation())
    non_anticipative._optimality_check.assert_called_once()
    non_anticipative._operational_check.assert_called_once()

def test_unfeasible_solution_in_check_optimality_and_feasibility_of_current_solution():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._operational_check = MagicMock()
    non_anticipative._is_opf_feasible = MagicMock(return_value=False)
    non_anticipative._optimality_check = MagicMock()
    non_anticipative._check_optimality_and_feasibility_of_current_solution(InterIterationInformation())
    non_anticipative._optimality_check.assert_not_called()
    non_anticipative._operational_check.assert_called_once()

def test_optimise_interventions_in_last_node():
    non_anticipative = NonAnticipativeRecursiveFunction()
    non_anticipative._get_available_interventions_for_current_year = MagicMock(return_value=[1, 2])
    non_anticipative._add_new_interventions_from_combinations = MagicMock()
    non_anticipative._check_optimality_and_feasibility_of_current_solution = MagicMock()
    non_anticipative._optimise_interventions_in_last_node(InterIterationInformation())
    non_anticipative._get_available_interventions_for_current_year.assert_called_once()
    assert non_anticipative._add_new_interventions_from_combinations.call_count == 3
    assert non_anticipative._check_optimality_and_feasibility_of_current_solution.call_count == 3
