from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, RecursiveFunction

class NonAnticipativeRecursiveFunction(RecursiveFunction):
    def _verify_feasibility_of_solution_in_successor_nodes(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        if self._verify_feasibility_of_sucessor_with_no_new_interventions(\
            inter_iteration_information):
            return True
        else:
            return self._verify_feasibility_of_sucessor_with_all_available_interventions_for_current_year(\
                inter_iteration_information)
    
    def _verify_feasibility_of_sucessor_with_no_new_interventions(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        self._operational_check(inter_iteration_information)
        return self._is_opf_feasible()
    
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

