from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, RecursiveFunction

class NonAnticipativeRecursiveFunction(RecursiveFunction):

    def solve(self, \
        inter_iteration_information: InterIterationInformation):
        # Optimality Check
        if not any(True for _ in self._control_graph.graph.neighbours(\
            inter_iteration_information.current_graph_node)):
            self._analisys_of_last_node_in_path(inter_iteration_information)
        self._exploration_of_current_solution(inter_iteration_information)
        self._interventions_handler(inter_iteration_information)

    def _interventions_handler(self, inter_iteration_information):
        _available_interventions = self._calculate_available_interventions(\
            inter_iteration_information)
        for number_combinations in range(1, len(_available_interventions) + 1):
            for combinations in self._calculate_all_combinations(\
                _available_interventions, number_combinations):
                self._add_new_interventions_from_combinations(inter_iteration_information, \
                    combinations)
                self._exploration_of_current_solution(inter_iteration_information)

    def _exploration_of_current_solution(self, inter_iteration_information):
        if self._verify_feasibility_of_solution_in_successor_nodes(inter_iteration_information):
            inter_iteration_information = self._construction_of_solution(\
                inter_iteration_information)
            self._graph_exploration(inter_iteration_information)

    def _analisys_of_last_node_in_path(self, inter_iteration_information):
        self._check_optimality_and_feasibility_of_current_solution(inter_iteration_information)
        self._optimise_interventions_in_last_node(inter_iteration_information)

    def _optimise_interventions_in_last_node(self, inter_iteration_information):
        all_available_interventions = self._get_available_interventions_for_current_year(\
            inter_iteration_information)
        for number_combinations in range(1, len(all_available_interventions) + 1):
            for combinations in self._calculate_all_combinations(\
                all_available_interventions, number_combinations):
                self._add_new_interventions_from_combinations(inter_iteration_information, \
                    combinations)
                self._check_optimality_and_feasibility_of_current_solution(inter_iteration_information)

    def _check_optimality_and_feasibility_of_current_solution(self, inter_iteration_information):
        self._operational_check(inter_iteration_information)
        if self._is_opf_feasible():
            self._optimality_check(inter_iteration_information)

    def _verify_feasibility_of_solution_in_successor_nodes(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        parent_node = inter_iteration_information.current_graph_node
        inter_iteration_information.level_in_graph += 1
        feasible = True
        for neighbour in self._control_graph.graph.neighbours(parent_node):
            inter_iteration_information.current_graph_node = neighbour
            if not self._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year(\
                    inter_iteration_information):
                feasible = False
                break
        inter_iteration_information.current_graph_node = parent_node
        inter_iteration_information.level_in_graph -= 1
        return feasible
    
    def _get_available_interventions_for_current_year(self, \
        inter_iteration_information: InterIterationInformation) -> AbstractDataContainer:
        viable_interventions = self._calculate_available_interventions(inter_iteration_information)
        for key, value in self._calculate_available_interventions(inter_iteration_information):
            if value.installation_time > 0:
                viable_interventions.pop(key)        
        return viable_interventions

    def _verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        all_available_interventions = self._get_available_interventions_for_current_year(\
            inter_iteration_information)
        inter_iteration_information.new_interventions.extend(all_available_interventions)
        self._operational_check(inter_iteration_information)
        for key, _ in all_available_interventions:
            inter_iteration_information.new_interventions.pop(key)
        return self._is_opf_feasible()

