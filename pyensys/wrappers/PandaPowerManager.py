from pyensys.wrappers.PandaPowerWrapper import PandaPowerWrapper
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData
from typing import List

from pyensys.wrappers.PandapowerDataClasses import SimulationSettings, Profile, OutputVariableSet, \
    TimeSeriesOutputFileSettings, UpdateParameterData


class PandaPowerManager:

    def __init__(self):
        self.wrapper = PandaPowerWrapper()
        self.simulation_settings = SimulationSettings()
        self._parameters = Parameters()

    def initialise_pandapower_network(self, opf_parameters: Parameters):
        self._parameters = opf_parameters
        self._initialise()

    def _initialise(self):
        self.load_mat_file_to_pandapower()
        self.add_controllers_to_network()
        self.add_output_writer_to_network()
        self.define_simulation_settings()
        self.create_static_generators_for_flexibility()

    def create_static_generators_for_flexibility(self):
        if len(self._parameters.optimisation_profiles_dataframes) > 0 and \
                self._parameters.optimisation_profiles_dataframes.get("flexible_units") is not None:
            generators_data = self._parameters.optimisation_profiles_dataframes.get("flexible_units")
            buses_indexes = list(generators_data["bus_index"].unique())
            for bus_index in buses_indexes:
                self.wrapper.create_poly_cost(index=self.wrapper.create_static_generator(bus_index),
                                              type_element="sgen")

    def load_mat_file_to_pandapower(self):
        if self._parameters.pandapower_mpc_settings.initialised:
            self.wrapper.load_mat_file_to_pandapower(
                file_path_with_extension=self._parameters.pandapower_mpc_settings.mat_file_path,
                frequency_hz=self._parameters.pandapower_mpc_settings.system_frequency)

    def add_controllers_to_network(self):
        if self._parameters.pandapower_profiles_data.initialised:
            profiles_pandapower = []
            for profile in self._parameters.pandapower_profiles_data.data:
                profiles_pandapower.append(Profile(
                    components_indexes_in_power_system=profile.indexes,
                    data=profile.data, column_names=profile.active_columns_names,
                    variable_name=profile.variable_name, components_type=profile.element_type))
            self.wrapper.add_controllers_to_network(profiles=profiles_pandapower)

    def add_output_writer_to_network(self):
        if self._parameters.output_settings.initialised and \
                self._parameters.opf_time_settings.initialised:
            output_variables = self._define_output_variables()
            output_settings = self._define_output_settings()
            self.wrapper.add_output_writer_to_network(output_settings, output_variables)

    def _define_output_variables(self) -> List[OutputVariableSet]:
        output_variables = []
        for variable in self._parameters.output_settings.output_variables:
            output_variables.append(OutputVariableSet(name_dataset=variable.name_dataset,
                                                      name_variable=variable.name_variable,
                                                      variable_indexes=variable.variable_indexes))
        return output_variables

    def _define_output_settings(self) -> TimeSeriesOutputFileSettings:
        output_settings = TimeSeriesOutputFileSettings(
            directory=self._parameters.output_settings.directory,
            number_time_steps=self._parameters.opf_time_settings.date_time_settings.size,
            format=self._parameters.output_settings.format)
        return output_settings

    def define_simulation_settings(self):
        if self._parameters.pandapower_optimisation_settings.initialised and \
                self._parameters.opf_time_settings.initialised and \
                self._parameters.problem_settings.initialised:
            self.simulation_settings = SimulationSettings(
                time_steps=list(range(self._parameters.opf_time_settings.date_time_settings.size)),
                display_progress_bar=
                self._parameters.pandapower_optimisation_settings.display_progress_bar,
                continue_on_divergence=
                self._parameters.pandapower_optimisation_settings.continue_on_divergence,
                optimisation_software=
                self._parameters.pandapower_optimisation_settings.optimisation_software,
                opf_type=self._parameters.problem_settings.opf_type)
        elif self._parameters.pandapower_optimisation_settings.initialised and \
                self._parameters.problem_settings.initialised:
            self.simulation_settings = SimulationSettings(
                time_steps=[],
                display_progress_bar=
                self._parameters.pandapower_optimisation_settings.display_progress_bar,
                continue_on_divergence=
                self._parameters.pandapower_optimisation_settings.continue_on_divergence,
                optimisation_software=
                self._parameters.pandapower_optimisation_settings.optimisation_software,
                opf_type=self._parameters.problem_settings.opf_type)

    def run_time_step_opf_pandapower(self):
        self.wrapper.run_time_step_simulation(self.simulation_settings)

    def update_network_controllers(self, pp_profiles_data: PandaPowerProfilesData):
        if pp_profiles_data.initialised:
            for profile in pp_profiles_data.data:
                pp_profile = Profile(
                    components_indexes_in_power_system=profile.indexes,
                    data=profile.data, column_names=profile.active_columns_names,
                    variable_name=profile.variable_name, components_type=profile.element_type)
                self.wrapper.update_network_controller(pp_profile)

    def is_feasible(self) -> bool:
        return self.wrapper.is_feasible()

    def update_parameter(self, parameter_data: UpdateParameterData):
        self.wrapper = PandaPowerWrapper()
        self._initialise()
        self.wrapper.network[parameter_data.component_type].at[
            parameter_data.parameter_position, parameter_data.parameter_name] = parameter_data.new_value

    def update_multiple_parameters(self, list_parameter_data: List[UpdateParameterData], initialise: bool = True):
        if initialise:
            self.wrapper = PandaPowerWrapper()
            self._initialise()
        for parameter in list_parameter_data:
            if parameter.component_type == "load":
                self.wrapper.network[parameter.component_type].at[
                    list(self.wrapper.network[parameter.component_type][
                        self.wrapper.network[parameter.component_type]["bus"] ==
                        parameter.parameter_position].index)[0],
                    parameter.parameter_name] = parameter.new_value
            else:
                self.wrapper.network[parameter.component_type].at[ \
                    parameter.parameter_position, parameter.parameter_name] = parameter.new_value

    def get_total_cost(self) -> float:
        return self.wrapper.get_total_cost()

    def run_ac_opf_pandapower(self):
        self.wrapper.run_ac_opf(self.simulation_settings)
