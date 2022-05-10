from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, RecursiveFunction, BinaryVariable, \
    UpdateParameterData, check_if_candidate_path_has_been_stored_in_incumbent

from typing import List
from copy import deepcopy


def _eliminate_offsprings_of_candidate_in_incumbent(info: InterIterationInformation):
    keys_to_eliminate = _find_keys_of_offsprings_in_incumbent(info)
    _delete_offsprings_from_incumbent(info, keys_to_eliminate)
    _renumber_keys_in_incumbent(info)
    if len(info.incumbent_interventions) == 0:
        info.allow_deleting_solutions_in_incumbent = False


def _find_keys_of_offsprings_in_incumbent(info: InterIterationInformation) -> List[str]:
    keys_to_eliminate = []
    for key, path in info.incumbent_graph_paths:
        if info.candidate_solution_path in path:
            keys_to_eliminate.append(key)
    return keys_to_eliminate


def _delete_offsprings_from_incumbent(info: InterIterationInformation, keys_to_eliminate: List[str]):
    for key in keys_to_eliminate:
        info.incumbent_interventions.pop(key)
        info.incumbent_graph_paths.pop(key)
        info.incumbent_operation_costs.pop(key)
        info.incumbent_investment_costs.pop(key)


def _renumber_keys_in_incumbent(info: InterIterationInformation):
    remaining_keys = _determine_keys_to_be_renumbered(info)
    for num, key in enumerate(remaining_keys):
        info.incumbent_interventions.append(str(num), info.incumbent_interventions.pop(key))
        info.incumbent_graph_paths.append(str(num), info.incumbent_graph_paths.pop(key))
        info.incumbent_operation_costs.append(str(num), info.incumbent_operation_costs.pop(key))
        info.incumbent_investment_costs.append(str(num), info.incumbent_investment_costs.pop(key))


def _determine_keys_to_be_renumbered(info: InterIterationInformation) -> List[str]:
    remaining_keys = []
    for key, _ in info.incumbent_graph_paths:
        remaining_keys.append(key)
    keys_to_keep = []
    for num, _ in enumerate(remaining_keys):
        if str(num) in remaining_keys:
            keys_to_keep.append(str(num))
    for key in keys_to_keep:
        if key in remaining_keys:
            remaining_keys.remove(key)
    return remaining_keys


def _add_new_interventions_from_combinations(inter_iteration_information: InterIterationInformation,
                                             combinations):
    inter_iteration_information.new_interventions = AbstractDataContainer()
    inter_iteration_information.new_interventions.create_list()
    inter_iteration_information.new_interventions_remaining_construction_time = AbstractDataContainer()
    inter_iteration_information.new_interventions_remaining_construction_time.create_list()
    for combination in combinations:
        for key, value in combination:
            inter_iteration_information.new_interventions.append(key, value)
            inter_iteration_information.new_interventions_remaining_construction_time.append(key,
                                                                                             value.installation_time)


def update_remaining_construction_time(info: InterIterationInformation, period: int = -1) -> \
        InterIterationInformation:
    if isinstance(info, InterIterationInformation):
        for _, interventions_time in info.candidate_interventions_remaining_construction_time:
            for key, _ in interventions_time:
                interventions_time[key] += period
        for key, _ in info.new_interventions_remaining_construction_time:
            info.new_interventions_remaining_construction_time[key] += period
        return info
    else:
        raise TypeError("The input is not of type InterIterationInformation")


def _get_already_build_interventions(info):
    candidate_interventions = deepcopy(info.candidate_interventions)
    for step, time_data in info.candidate_interventions_remaining_construction_time:
        elements_to_remove = []
        for key, time in time_data:
            if time > 0:
                elements_to_remove.append(key)
        for key in elements_to_remove:
            candidate_interventions[step].pop(key)
    return candidate_interventions


def _get_constructed_interventions_from_candidate(info: InterIterationInformation) -> List[BinaryVariable]:
    accepted_interventions = []
    for (_, interventions_time), (_, interventions) in zip(info.candidate_interventions_remaining_construction_time,
                                                           info.candidate_interventions):
        for key, time in interventions_time:
            if time <= 0:
                accepted_interventions.append(interventions[key])
    return accepted_interventions


def _get_constructed_interventions_from_new_interventions(info: InterIterationInformation):
    accepted_interventions = []
    for (_, time), (_, intervention) in zip(info.new_interventions_remaining_construction_time,
                                            info.new_interventions):
        if time <= 0:
            accepted_interventions.append(intervention)
    return accepted_interventions


def _get_interventions_ready_to_operate_in_opf(info: InterIterationInformation) -> List[BinaryVariable]:
    interventions = _get_constructed_interventions_from_candidate(info)
    interventions.extend(_get_constructed_interventions_from_new_interventions(info))
    return interventions


def _append_candidate_interventions_in_incumbent_interventions_list(key_in_incumbent: str,
                                                                    info: InterIterationInformation) -> \
        InterIterationInformation:
    info.incumbent_interventions.append(key_in_incumbent, _get_already_build_interventions(info))
    return info


def _replace_incumbent_interventions_list_with_candidate_interventions(key_in_incumbent: str,
                                                                       info: InterIterationInformation) -> \
        InterIterationInformation:
    info.incumbent_interventions[key_in_incumbent] = _get_already_build_interventions(info)
    return info


def _calculate_total_planning_cost(investment_data: AbstractDataContainer,
                                   operation_data: AbstractDataContainer) -> float:
    cost = 0.0
    for (_, investment), (_, operation) in zip(investment_data, operation_data):
        cost += investment + operation
    return cost


def _store_complete_tree(info):
    info.complete_tree.interventions = info.incumbent_interventions
    info.complete_tree.graph_paths = info.incumbent_graph_paths
    info.complete_tree.investment_costs = info.incumbent_investment_costs
    info.complete_tree.operation_costs = info.incumbent_operation_costs


def _replacement_of_investments_for_whole_tree(info: InterIterationInformation):
    if info.complete_tree.is_empty():
        _store_complete_tree(info)
    else:
        cost_existing_tree = _calculate_total_planning_cost(info.complete_tree.investment_costs,
                                                            info.complete_tree.operation_costs)
        cost_candidate_tree = _calculate_total_planning_cost(info.incumbent_investment_costs,
                                                             info.incumbent_operation_costs)
        if cost_existing_tree < cost_candidate_tree:
            _store_complete_tree(info)


class NonAnticipativeRecursiveFunction(RecursiveFunction):
    def __init__(self):
        super().__init__()
        self._tree_nodes = [self._control_graph.graph]

    def solve(self, info: InterIterationInformation) -> bool:
        info = update_remaining_construction_time(info)
        if not any(True for _ in self._control_graph.graph.neighbours(info.current_graph_node)):
            if self._check_feasibility_of_current_solution(info):
                self._optimality_check(info)
            self._optimise_interventions_in_last_node(info)
        self._exploration_of_current_solution(info)
        feasible_solution_exist = self._interventions_handler(info)
        info = update_remaining_construction_time(info, 1)
        return feasible_solution_exist

    def _interventions_handler(self, inter_iteration_information: InterIterationInformation) -> bool:
        _available_interventions = self._calculate_available_interventions(inter_iteration_information)
        feasible_solution_exist = False
        for number_combinations in range(1, len(_available_interventions) + 1):
            for combinations in self._calculate_all_combinations(_available_interventions, number_combinations):
                _add_new_interventions_from_combinations(inter_iteration_information, combinations)
                if self._exploration_of_current_solution(inter_iteration_information):
                    feasible_solution_exist = True
        return feasible_solution_exist

    def _exploration_of_current_solution(self, inter_iteration_information: InterIterationInformation):
        if self._check_feasibility_of_current_solution(inter_iteration_information) and \
                self._verify_feasibility_of_solution_in_successor_nodes(inter_iteration_information):
            inter_iteration_information = self._construction_of_solution(inter_iteration_information)
            return self._graph_exploration(inter_iteration_information)
        else:
            return False

    def _graph_exploration(self, info: InterIterationInformation) -> bool:
        feasible_solution_exist = self._exploration_of_successors(info)
        if not feasible_solution_exist or info.allow_deleting_solutions_in_incumbent:
            _eliminate_offsprings_of_candidate_in_incumbent(info)
        return feasible_solution_exist

    def _exploration_of_successors(self, info: InterIterationInformation) -> bool:
        parent_node = info.current_graph_node
        feasible_solution_exist = True
        info.level_in_graph += 1
        for neighbour in self._control_graph.graph.neighbours(info.current_graph_node):
            info.current_graph_node = neighbour
            if not self.solve(info=info):
                feasible_solution_exist = False
                break
        info.current_graph_node = parent_node
        info.level_in_graph -= 1
        return feasible_solution_exist

    def _optimise_interventions_in_last_node(self, inter_iteration_information: InterIterationInformation):
        all_available_interventions = self._get_available_interventions_for_current_year(inter_iteration_information)
        for number_combinations in range(1, len(all_available_interventions) + 1):
            for combinations in self._calculate_all_combinations(all_available_interventions, number_combinations):
                _add_new_interventions_from_combinations(inter_iteration_information, combinations)
                if self._check_feasibility_of_current_solution(inter_iteration_information):
                    self._optimality_check(inter_iteration_information)

    def _check_feasibility_of_current_solution(self, info: InterIterationInformation) -> bool:
        self._operational_check(info)
        return self._is_opf_feasible()

    def _verify_feasibility_of_solution_in_successor_nodes(self,
                                                           inter_iteration_information: InterIterationInformation) -> \
            bool:
        parent_node = inter_iteration_information.current_graph_node
        inter_iteration_information.level_in_graph += 1
        feasible = True
        for neighbour in self._control_graph.graph.neighbours(parent_node):
            inter_iteration_information.current_graph_node = neighbour
            if not self._verify_feasibility_of_successor_with_all_available_interventions_for_current_year(
                    inter_iteration_information):
                feasible = False
                break
        inter_iteration_information.current_graph_node = parent_node
        inter_iteration_information.level_in_graph -= 1
        return feasible

    def _get_available_interventions_for_current_year(self, inter_iteration_information: InterIterationInformation) -> \
            AbstractDataContainer:
        viable_interventions = self._calculate_available_interventions(inter_iteration_information)
        for key, value in self._calculate_available_interventions(inter_iteration_information):
            if value.installation_time > 0:
                viable_interventions.pop(key)
        return viable_interventions

    def _verify_feasibility_of_successor_with_all_available_interventions_for_current_year(self,
                                                                                           inter_iteration_information:
                                                                                           InterIterationInformation) \
            -> bool:
        all_available_interventions = self._get_available_interventions_for_current_year(inter_iteration_information)
        inter_iteration_information.new_interventions.extend(all_available_interventions)
        self._operational_check(inter_iteration_information)
        for key, _ in all_available_interventions:
            inter_iteration_information.new_interventions.pop(key)
        return self._is_opf_feasible()

    def _create_list_of_parameters_to_update_in_opf(self, variables: List[BinaryVariable]) -> \
            List[UpdateParameterData]:
        parameters = []
        for variable in variables:
            parameters.append(self._create_data_to_update_parameter(variable))
        return parameters

    def _update_status_elements_opf(self, inter_iteration_information: InterIterationInformation):
        if len(inter_iteration_information.new_interventions) > 0:
            self._opf.update_multiple_parameters(
                self._create_list_of_parameters_to_update_in_opf(
                    _get_interventions_ready_to_operate_in_opf(inter_iteration_information)))

    def _append_candidate_in_incumbent_list(self, key_in_incumbent: str,
                                            info: InterIterationInformation) -> \
            InterIterationInformation:
        info.incumbent_graph_paths.append(key_in_incumbent, deepcopy(info.candidate_solution_path))
        info = _append_candidate_interventions_in_incumbent_interventions_list(key_in_incumbent, info)
        info.incumbent_investment_costs.append(key_in_incumbent,
                                               self._calculate_investment_cost(info.candidate_interventions))
        info.incumbent_operation_costs.append(key_in_incumbent,
                                              self._calculate_opteration_cost(info.candidate_operation_cost))
        return info

    def _replace_incumbent_if_candidate_is_better(self, key_in_incumbent: str,
                                                  info: InterIterationInformation):
        total_cost_candidate = self._calculate_investment_cost(_get_already_build_interventions(info)) + \
                               self._calculate_opteration_cost(info.candidate_operation_cost)
        total_cost_incumbent = info.incumbent_investment_costs[key_in_incumbent] + \
                               info.incumbent_operation_costs[key_in_incumbent]
        if total_cost_incumbent > total_cost_candidate:
            self._replace_solution_in_incumbent_list(key_in_incumbent, info)

    def _replace_solution_in_incumbent_list(self, key_in_incumbent: str,
                                            info: InterIterationInformation):
        info.incumbent_graph_paths[key_in_incumbent] = deepcopy(info.candidate_solution_path)
        info = _replace_incumbent_interventions_list_with_candidate_interventions(key_in_incumbent, info)
        info.incumbent_investment_costs[key_in_incumbent] = \
            self._calculate_investment_cost(_get_already_build_interventions(info))
        info.incumbent_operation_costs[key_in_incumbent] = \
            self._calculate_opteration_cost(info.candidate_operation_cost)

    def _optimality_check_with_non_empty_incumbent(self, info):
        key_path = check_if_candidate_path_has_been_stored_in_incumbent(info)
        if key_path == "not found":
            self._append_candidate_in_incumbent_list(str(len(info.incumbent_graph_paths)), info)
        else:
            self._replace_incumbent_if_candidate_is_better(key_path, info)
        if self._check_if_all_nodes_in_tree_have_been_visited(info):
            info.allow_deleting_solutions_in_incumbent = True
            _replacement_of_investments_for_whole_tree(info)

    def _check_if_all_nodes_in_tree_have_been_visited(self, info: InterIterationInformation) -> bool:
        if len(self._determine_remaining_nodes_to_be_explored(info)) == 0:
            return True
        else:
            return False

    def _determine_remaining_nodes_to_be_explored(self, info):
        remaining_nodes = deepcopy(self._tree_nodes)
        for _, path in info.incumbent_graph_paths:
            for _, node in path:
                if node in remaining_nodes:
                    remaining_nodes.remove(node)
        return remaining_nodes
