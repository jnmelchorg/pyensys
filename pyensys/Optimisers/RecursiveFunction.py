from pyensys.wrappers.PandaPowerManager import PandaPowerManager, UpdateParameterData
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData, \
    PandaPowerProfileData
from pyensys.Optimisers.ControlGraphsCreator import GraphandClusterData, ClusterData, \
    RecursiveFunctionGraphCreator
from pyensys.DataContainersInterface.AbstractDataContainer import AbstractDataContainer
from pyensys.DataContainersInterface.OperationsAbstractDataContainer import difference_abstract_data_containers

from dataclasses import dataclass, field
from itertools import combinations
from copy import deepcopy
from typing import Iterator, Tuple


@dataclass
class InvestmentPlanData:
    interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    operation_costs: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    investment_costs: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    graph_paths: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())

    def _assert_all_data_has_same_length(self):
        assert len(self.interventions) == len(self.operation_costs) == len(self.investment_costs) == \
               len(self.graph_paths), "All data must have the same length"

    def is_empty(self):
        self._assert_all_data_has_same_length()
        return len(self.interventions) == 0

    def __len__(self):
        self._assert_all_data_has_same_length()
        return len(self.interventions)


@dataclass
class InterIterationInformation:
    incumbent_interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    incumbent_operation_costs: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    incumbent_investment_costs: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    incumbent_graph_paths: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    candidate_solution_path: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    candidate_operation_cost: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    candidate_interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    candidate_interventions_remaining_construction_time: \
        AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    new_interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    new_interventions_remaining_construction_time: \
        AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    complete_tree: InvestmentPlanData = \
        field(default_factory=lambda: InvestmentPlanData())
    partial_tree: InvestmentPlanData = \
        field(default_factory=lambda: InvestmentPlanData())
    current_graph_node: int = 0
    level_in_graph: int = 0
    allow_deleting_solutions_in_incumbent: bool = False
    last_node_reached: bool = False


@dataclass
class BinaryVariable:
    element_id: str = ""
    cost: float = 0.0
    element_position: int = 0
    element_type: str = ""
    variable_name: str = ""
    installation_time: int = -1

    def __eq__(self, other):
        if isinstance(other, BinaryVariable):
            return (self.element_id == other.element_id) and (self.cost == other.cost) and \
                   (self.element_position == other.element_position) and \
                   (self.element_type == other.element_type) and (self.variable_name == other.variable_name)


@dataclass
class InterventionsInformation:
    interventions: AbstractDataContainer = \
        field(default_factory=lambda: AbstractDataContainer())
    path: str = ''
    year: int = 0


def check_if_candidate_path_has_been_stored_in_incumbent(inter_iteration_information: InterIterationInformation) -> \
        str:
    for key, value in inter_iteration_information.incumbent_graph_paths:
        if inter_iteration_information.candidate_solution_path in value:
            return key
    return "not found"


def _create_data_to_update_status_of_parameter(intervention: BinaryVariable) -> UpdateParameterData:
    if isinstance(intervention, BinaryVariable):
        parameter_to_update = UpdateParameterData()
        parameter_to_update.component_type = intervention.element_type
        parameter_to_update.parameter_name = "in_service"
        parameter_to_update.parameter_position = intervention.element_position
        parameter_to_update.new_value = True
        return parameter_to_update
    else:
        raise TypeError


def _create_load_parameter(row, parameter_name: str):
    parameter_to_update = UpdateParameterData()
    parameter_to_update.component_type = "load"
    parameter_to_update.parameter_position = int(row["bus_index"])
    parameter_to_update.parameter_name = parameter_name
    parameter_to_update.new_value = float(row[parameter_name])
    return parameter_to_update


class RecursiveFunction:

    def __init__(self):
        self._parameters = Parameters()
        self._control_graph = GraphandClusterData()
        self._pool_interventions = AbstractDataContainer()
        self._opf = PandaPowerManager()
        self.__DAYS_PER_YEAR = 365
        self._replace_parameters_individually = False

    def initialise(self, parameters: Parameters):
        self._parameters = parameters
        if len(self._parameters.optimisation_profiles_dataframes) > 0:
            self._replace_parameters_individually = True
        self._create_control_graph()
        self._create_pool_interventions()
        if self._parameters.problem_settings.opf_optimizer == "pandapower":
            self._initialise_pandapower()

    def _initialise_pandapower(self):
        self._opf.initialise_pandapower_network(self._parameters)
        self.original_pp_profiles_data = self._parameters.pandapower_profiles_data

    def _create_control_graph(self):
        control_graph = RecursiveFunctionGraphCreator()
        self._control_graph = control_graph.create_recursive_function_graph(self._parameters)

    def _create_pool_interventions(self):
        self._pool_interventions.create_list()
        counter = 0
        for variables in self._parameters.optimisation_binary_variables:
            for cost, position, time in zip(variables.costs, variables.elements_positions, variables.installation_time):
                self._pool_interventions.append(str(counter), BinaryVariable(
                    element_type=variables.element_type, variable_name=variables.variable_name,
                    element_id='', element_position=position, cost=cost, installation_time=time))
                counter += 1

    def solve(self, inter_iteration_information: InterIterationInformation):
        self._operational_check(inter_iteration_information)
        if self._is_opf_feasible():
            inter_iteration_information = self._construction_of_solution(inter_iteration_information)
            # Optimality Check
            if not any(True for _ in self._control_graph.graph.neighbours(
                    inter_iteration_information.current_graph_node)):
                self._optimality_check(inter_iteration_information)
                return
            # Intervention Handler
            self._interventions_handler(inter_iteration_information)

    def _interventions_handler(self, inter_iteration_information: InterIterationInformation):
        _available_interventions = self._calculate_available_interventions( \
            inter_iteration_information)
        self._graph_exploration(inter_iteration_information)
        for number_combinations in range(1, len(_available_interventions) + 1):
            for combinations in self._calculate_all_combinations( \
                    _available_interventions, number_combinations):
                self._add_new_interventions_from_combinations(inter_iteration_information, \
                                                              combinations)
                # Graph exploration
                self._graph_exploration(inter_iteration_information)

    def _add_new_interventions_from_combinations(self, inter_iteration_information: InterIterationInformation,
                                                 combination):
        inter_iteration_information.new_interventions = AbstractDataContainer()
        inter_iteration_information.new_interventions.create_list()
        for element in combination:
            inter_iteration_information.new_interventions.extend(element)

    def _graph_exploration(self, inter_iteration_information: InterIterationInformation):
        parent_node = inter_iteration_information.current_graph_node
        for neighbour in self._control_graph.graph.neighbours(inter_iteration_information.current_graph_node):
            inter_iteration_information.level_in_graph += 1
            inter_iteration_information.current_graph_node = neighbour
            self.solve(inter_iteration_information=inter_iteration_information)
        inter_iteration_information.current_graph_node = parent_node

    def _operational_check(self, inter_iteration_information: InterIterationInformation):
        self._update_status_elements_opf(inter_iteration_information)
        if self._parameters.problem_settings.inter_temporal:
            self._update_pandapower_controllers(inter_iteration_information.current_graph_node)
        if self._replace_parameters_individually:
            self._update_parameters_in_opf(inter_iteration_information)
        self._run_opf()

    def _update_status_elements_opf(self, inter_iteration_information: InterIterationInformation):
        if len(inter_iteration_information.new_interventions) > 0:
            self._update_status_elements_opf_per_intervention_group(inter_iteration_information.candidate_interventions)
            self._update_status_elements_opf_per_intervention_group(inter_iteration_information.new_interventions)
        else:
            return

    def _update_status_elements_opf_per_intervention_group(self, interventions: AbstractDataContainer):
        parameters_to_update = []
        for _, intervention in interventions:
            if isinstance(intervention, BinaryVariable):
                parameters_to_update.append(_create_data_to_update_status_of_parameter(intervention))
            elif isinstance(intervention, AbstractDataContainer):
                for _, value in intervention:
                    parameters_to_update.append(_create_data_to_update_status_of_parameter(value))
            else:
                raise TypeError
        self._opf.update_multiple_parameters(parameters_to_update)

    def _update_pandapower_controllers(self, current_graph_node: int):
        new_profiles = self._create_new_pandapower_profiles(current_graph_node)
        self._opf.update_network_controllers(new_profiles)

    def _run_opf(self):
        if self._parameters.problem_settings.opf_optimizer == "pandapower" and \
                self._parameters.problem_settings.inter_temporal:
            self._opf.run_time_step_opf_pandapower()
        elif self._parameters.problem_settings.opf_optimizer == "pandapower":
            self._opf.run_ac_opf_pandapower()

    def _construction_of_solution(self, inter_iteration_information: InterIterationInformation) \
            -> InterIterationInformation:
        inter_iteration_information.candidate_solution_path.append(str(len(
            inter_iteration_information.candidate_solution_path)),
            inter_iteration_information.current_graph_node)
        inter_iteration_information.candidate_interventions.append(
            str(inter_iteration_information.level_in_graph),
            deepcopy(inter_iteration_information.new_interventions))
        inter_iteration_information.candidate_interventions_remaining_construction_time.append(
            str(inter_iteration_information.level_in_graph),
            deepcopy(inter_iteration_information.new_interventions_remaining_construction_time))
        inter_iteration_information.candidate_operation_cost.append(
            str(inter_iteration_information.level_in_graph),
            self._get_total_operation_cost())
        return inter_iteration_information

    def _calculate_available_interventions(self, inter_iteration_information: InterIterationInformation) -> \
            AbstractDataContainer:
        _available_interventions = deepcopy(self._pool_interventions)
        for _, previous in inter_iteration_information.candidate_interventions:
            _available_interventions = difference_abstract_data_containers(
                _available_interventions, previous)
        _available_interventions = difference_abstract_data_containers(
            _available_interventions,
            inter_iteration_information.new_interventions)
        return _available_interventions

    def _calculate_all_combinations(self, available_interventions: AbstractDataContainer, length_set: int) -> \
            Iterator[Tuple[int]]:
        return combinations(available_interventions, length_set)

    def _create_new_pandapower_profiles(self, current_graph_node: int) \
            -> PandaPowerProfilesData:
        new_profiles = PandaPowerProfilesData(initialised=True)
        for modifier in self._control_graph.nodes_data[current_graph_node]:
            new_profiles.data.append(self._create_new_pandapower_profile(modifier))
        return new_profiles

    def _create_new_pandapower_profile(self, modifier_info: ClusterData) -> PandaPowerProfileData:
        profile = deepcopy(self.original_pp_profiles_data.data[self._get_profile_position_to_update(modifier_info)])
        profile.data = profile.data * modifier_info.centroid
        return profile

    def _get_profile_position_to_update(self, modifier_info: ClusterData) -> int:
        for position, pp_profile in enumerate(self.original_pp_profiles_data.data):
            if modifier_info.element_type == pp_profile.element_type and \
                    modifier_info.variable_name == pp_profile.variable_name:
                return position
        return -1

    def _is_opf_feasible(self):
        return self._opf.is_feasible()

    def _get_total_operation_cost(self):
        return self._opf.get_total_cost() * self.__DAYS_PER_YEAR

    def _optimality_check(self, inter_iteration_information: InterIterationInformation):
        if len(inter_iteration_information.incumbent_graph_paths) == 0:
            self._append_candidate_in_incumbent_list("0", inter_iteration_information)
        else:
            self._optimality_check_with_non_empty_incumbent(inter_iteration_information)
        self._return_to_previous_state(inter_iteration_information)

    def _optimality_check_with_non_empty_incumbent(self, inter_iteration_information):
        key_path = check_if_candidate_path_has_been_stored_in_incumbent(inter_iteration_information)
        if key_path == "not found":
            self._append_candidate_in_incumbent_list(str(len(inter_iteration_information.incumbent_graph_paths)),
                                                     inter_iteration_information)
        else:
            self._replace_incumbent_if_candidate_is_better(key_path, inter_iteration_information)

    def _replace_incumbent_if_candidate_is_better(self, key_in_incumbent: str,
                                                  inter_iteration_information: InterIterationInformation):
        total_cost_candidate = self._calculate_investment_cost(inter_iteration_information.candidate_interventions) + \
                               self._calculate_opteration_cost(inter_iteration_information.candidate_operation_cost)
        total_cost_incumbent = inter_iteration_information.incumbent_investment_costs[key_in_incumbent] + \
                               inter_iteration_information.incumbent_operation_costs[key_in_incumbent]
        if total_cost_incumbent > total_cost_candidate:
            self._replace_solution_in_incumbent_list(key_in_incumbent, inter_iteration_information)

    def _replace_solution_in_incumbent_list(self, key_in_incumbent: str,
                                            inter_iteration_information: InterIterationInformation):
        inter_iteration_information.incumbent_graph_paths[key_in_incumbent] = \
            deepcopy(inter_iteration_information.candidate_solution_path)
        inter_iteration_information.incumbent_interventions[key_in_incumbent] = \
            deepcopy(inter_iteration_information.candidate_interventions)
        inter_iteration_information.incumbent_investment_costs[key_in_incumbent] = \
            self._calculate_investment_cost(inter_iteration_information.candidate_interventions)
        inter_iteration_information.incumbent_operation_costs[key_in_incumbent] = \
            self._calculate_opteration_cost(inter_iteration_information.candidate_operation_cost)

    def _append_candidate_in_incumbent_list(self, key_in_incumbent: str,
                                            inter_iteration_information: InterIterationInformation):
        inter_iteration_information.incumbent_graph_paths.append(key_in_incumbent, deepcopy(
            inter_iteration_information.candidate_solution_path))
        inter_iteration_information.incumbent_interventions.append(key_in_incumbent, deepcopy(
            inter_iteration_information.candidate_interventions))
        inter_iteration_information.incumbent_investment_costs. \
            append(key_in_incumbent,
                   self._calculate_investment_cost(inter_iteration_information.candidate_interventions))
        inter_iteration_information.incumbent_operation_costs. \
            append(key_in_incumbent,
                   self._calculate_opteration_cost(inter_iteration_information.candidate_operation_cost))

    def _return_to_previous_state(self, inter_iteration_information: InterIterationInformation):
        inter_iteration_information.candidate_interventions.pop(
            str(inter_iteration_information.level_in_graph))
        inter_iteration_information.candidate_interventions_remaining_construction_time.pop(
            str(inter_iteration_information.level_in_graph))
        inter_iteration_information.candidate_operation_cost.pop(
            str(inter_iteration_information.level_in_graph))
        inter_iteration_information.candidate_solution_path.pop(str(len(
            inter_iteration_information.candidate_solution_path) - 1))
        if not self._parameters.problem_settings.non_anticipative:
            inter_iteration_information.level_in_graph -= 1

    def _calculate_investment_cost(self, interventions: AbstractDataContainer) -> float:
        total_cost = 0.0
        for year, (_, investments) in enumerate(interventions):
            partial_cost = 0.0
            for _, investment in investments:
                partial_cost = partial_cost + investment.cost
            total_cost = total_cost + (1 / (
                        (1 - (self._parameters.problem_settings.return_rate_in_percentage / 100)) ** year)) * \
                         partial_cost
        return total_cost

    def _calculate_opteration_cost(self, operation_cost_per_year: AbstractDataContainer) -> float:
        total_cost = 0.0
        for year, (_, operation_cost) in enumerate(operation_cost_per_year):
            total_cost = total_cost + (1 / (
                        (1 - (self._parameters.problem_settings.return_rate_in_percentage / 100)) ** year)) * \
                         operation_cost
        return total_cost

    def _update_parameters_in_opf(self, info: InterIterationInformation):
        parameters_to_update = []
        for name, row in self._control_graph.map_node_to_data_power_system[info.current_graph_node]["buses"].iterrows():
            if "p_mw" in row.index:
                parameter_to_update = _create_load_parameter(row, "p_mw")
                parameters_to_update.append(parameter_to_update)
            if "q_mvar" in row.index:
                parameter_to_update = _create_load_parameter(row, "q_mvar")
                parameters_to_update.append(parameter_to_update)
        self._opf.update_multiple_parameters(parameters_to_update, False)
