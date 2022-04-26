from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, RecursiveFunction, BinaryVariable

from typing import List


def _eliminate_offsprings_of_candidate_in_incumbent(info: InterIterationInformation):
    keys_to_eliminate = _find_keys_of_offsprings_in_incumbent(info)
    _delete_offsprings_from_incumbent(info, keys_to_eliminate)
    _renumber_keys_in_incumbent(info)


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
    remaining_keys = []
    for key, _ in info.incumbent_graph_paths:
        remaining_keys.append(key)
    for num, key in enumerate(remaining_keys):
        info.incumbent_interventions.append(str(num), info.incumbent_interventions.pop(key))
        info.incumbent_graph_paths.append(str(num), info.incumbent_graph_paths.pop(key))
        info.incumbent_operation_costs.append(str(num), info.incumbent_operation_costs.pop(key))
        info.incumbent_investment_costs.append(str(num), info.incumbent_investment_costs.pop(key))


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


def update_remaining_construction_time(info: InterIterationInformation) -> \
        InterIterationInformation:
    if isinstance(info, InterIterationInformation):
        for _, interventions_time in info.candidate_interventions_remaining_construction_time:
            for key, _ in interventions_time:
                interventions_time[key] -= 1
        for key, _ in info.new_interventions_remaining_construction_time:
            info.new_interventions_remaining_construction_time[key] -= 1
        return info
    else:
        raise TypeError("The input is not of type InterIterationInformation")


class NonAnticipativeRecursiveFunction(RecursiveFunction):

    def solve(self, inter_iteration_information: InterIterationInformation) -> bool:
        inter_iteration_information = update_remaining_construction_time(inter_iteration_information)
        if not any(True for _ in self._control_graph.graph.neighbours(inter_iteration_information.current_graph_node)):
            if self._check_feasibility_of_current_solution(inter_iteration_information):
                self._optimality_check(inter_iteration_information)
            self._optimise_interventions_in_last_node(inter_iteration_information)
        self._exploration_of_current_solution(inter_iteration_information)
        return self._interventions_handler(inter_iteration_information)

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

    def _graph_exploration(self, inter_iteration_information: InterIterationInformation) -> bool:
        parent_node = inter_iteration_information.current_graph_node
        feasible_solution_exist = True
        inter_iteration_information.level_in_graph += 1
        for neighbour in self._control_graph.graph.neighbours(inter_iteration_information.current_graph_node):
            inter_iteration_information.current_graph_node = neighbour
            if not self.solve(inter_iteration_information=inter_iteration_information):
                feasible_solution_exist = False
                break
        inter_iteration_information.current_graph_node = parent_node
        inter_iteration_information.level_in_graph -= 1
        if not feasible_solution_exist:
            _eliminate_offsprings_of_candidate_in_incumbent(inter_iteration_information)
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
