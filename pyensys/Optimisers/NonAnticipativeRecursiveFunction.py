from pyensys.Optimisers.RecursiveFunction import InterIterationInformation, RecursiveFunction

from  typing import List

class NonAnticipativeRecursiveFunction(RecursiveFunction):
    def _verify_feasibility_of_solution_in_successor_nodes(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        if self._verify_feasibility_of_sucessor_with_no_new_interventions(\
            inter_iteration_information):
            return True
        else:
            pass
    
    def _verify_feasibility_of_sucessor_with_no_new_interventions(self, \
        inter_iteration_information: InterIterationInformation) -> bool:
        self._operational_check(inter_iteration_information)
        return self._is_opf_feasible()
    
    def _get_available_interventions_for_current_year(self, \
        inter_iteration_information: InterIterationInformation) -> List[str]:
        return []
