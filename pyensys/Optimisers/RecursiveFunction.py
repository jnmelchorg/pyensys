from pyensys.wrappers.PandaPowerManager import PandaPowerManager
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData, \
    PandaPowerProfileData
from pyensys.Optimisers.ControlGraphsCreator import ControlGraphData, ClusterData, \
    RecursiveFunctionGraphCreator
from pyensys.AbstractDataContainer import AbstractDataContainer

from typing import List, Any
from dataclasses import dataclass, field

from copy import copy

@dataclass
class InterIterationInformation:
    incumbent_interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    incumbent_graph_paths: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    partial_solution_interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    partial_solution_operation_cost: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    partial_solution_path: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    current_graph_node: int = 0

@dataclass
class BinaryVariable:
    element_id: str = ""
    cost: float = 0.0
    element_position: int = 0
    element_type: str = ""
    variable_name: str = ""

class RecursiveFunction:

    def __init__(self):
        self._parameters = Parameters()
        self._control_graph = ControlGraphData()
        self._node_under_analysis: int = -1
        self._inter_iteration_information = InterIterationInformation()

    def _operational_check(self):
        if self._parameters.problem_settings.opf_optimizer == "pandapower" and \
            self._parameters.problem_settings.intertemporal:
            self.pp_opf.run_timestep_opf_pandapower()
        
    def initialise(self, parameters: Parameters):
        self._parameters = parameters
        self._create_control_graph()
        self._create_pool_interventions()
        if self._parameters.problem_settings.opf_optimizer == "pandapower":
            self._initialise_pandapower()
    
    def _initialise_pandapower(self):
        self.pp_opf = PandaPowerManager()
        self.pp_opf.initialise_pandapower_network(self._parameters)
        self.original_pp_profiles_data = self._parameters.pandapower_profiles_data

    def _create_control_graph(self):
        control_graph = RecursiveFunctionGraphCreator()
        self._control_graph = control_graph.create_recursive_function_graph(self._parameters)
    
    def _create_pool_interventions(self):
        pass

    def solve(self, inter_iteration_information: InterIterationInformation):
        self._node_under_analysis = copy(inter_iteration_information.current_graph_node)
        self._update_pandapower_controllers()
        self._operational_check()
        if self.pp_opf.is_feasible():
            is_end_node = True
            for neighbour in self._control_graph.graph.neighbors(self._node_under_analysis):
                is_end_node = False
                inter_iteration_information.current_graph_node = neighbour
                self.solve(inter_iteration_information=inter_iteration_information)
            if is_end_node:
                self._optimality_check()

    def _optimality_check(self):
        pass

    def _calculate_interventions_cost(self):
        pass
    
    def _update_pandapower_controllers(self):
        new_profiles = self._create_new_pandapower_profiles()
        self.pp_opf.update_network_controllers(new_profiles)
    
    def _create_new_pandapower_profiles(self) -> PandaPowerProfilesData:
        new_profiles = PandaPowerProfilesData(initialised=True)
        for modifier in self._control_graph.nodes_data[self._node_under_analysis]:
            new_profiles.data.append(self._create_new_pandapower_profile(modifier))
        return new_profiles
    
    def _create_new_pandapower_profile(self, modifier_info: ClusterData) -> PandaPowerProfileData:
        profile = copy(self.original_pp_profiles_data.data[\
            self._get_profile_position_to_update(modifier_info)])
        profile.data = profile.data * modifier_info.centroid
        return profile
    
    def _get_profile_position_to_update(self, modifier_info: ClusterData) -> int:
        for position, pp_profile in enumerate(self.original_pp_profiles_data.data):
            if modifier_info.element_type == pp_profile.element_type and \
                modifier_info.variable_name == pp_profile.variable_name:
                return position
        return -1


