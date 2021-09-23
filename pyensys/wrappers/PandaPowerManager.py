from pyensys.wrappers.PandapowerDataClasses import *
from pyensys.wrappers.PandaPowerWrapper import PandaPowerWrapper
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfilesData
from typing import List

class PandaPowerManager():

    def __init__(self):
        self.wrapper = PandaPowerWrapper()
        self.simulation_settings = SimulationSettings()

    def initialise_pandapower_network(self, problem_parameters: Parameters) -> PandaPowerWrapper:
        self.load_mat_file_to_pandapower(problem_parameters)
        self.add_controllers_to_network(problem_parameters)
        self.add_output_writer_to_network(problem_parameters)
        self.define_simulation_settings(problem_parameters)

    def load_mat_file_to_pandapower(self, problem_parameters: Parameters):
        if problem_parameters.pandapower_mpc_settings.initialised:
            self.wrapper.load_mat_file_to_pandapower(\
                filename_with_extension=problem_parameters.pandapower_mpc_settings.mat_file_path, \
                frequency_hz=problem_parameters.pandapower_mpc_settings.system_frequency)
    
    def add_controllers_to_network(self, problem_parameters: Parameters):
        if problem_parameters.pandapower_profiles_data.initialised:
            profiles_pandapower = []
            for profile in problem_parameters.pandapower_profiles_data.data:
                profiles_pandapower.append(Profile(\
                    components_indexes_in_power_system=profile.indexes, \
                    data=profile.data, column_names=profile.active_columns_names, \
                    variable_name=profile.variable_name, components_type=profile.element_type))
            self.wrapper.add_controllers_to_network(profiles=profiles_pandapower)

    def add_output_writer_to_network(self, problem_parameters: Parameters):
        if problem_parameters.output_settings.initialised and \
            problem_parameters.opf_time_settings.initialised:
            output_variables = self._define_output_variables(problem_parameters)
            output_settings = self._define_output_settings(problem_parameters)
            self.wrapper.add_output_writer_to_network(output_settings, output_variables)
    
    def _define_output_variables(self, problem_parameters: Parameters) -> List[OutputVariableSet]:
        output_variables = []
        for variable in problem_parameters.output_settings.output_variables:
            output_variables.append(OutputVariableSet(name_dataset=variable.name_dataset, \
                name_variable=variable.name_variable, variable_indexes=variable.variable_indexes))
        return output_variables
    
    def _define_output_settings(self, problem_parameters: Parameters) -> TimeSeriesOutputFileSettings:
        output_settings = TimeSeriesOutputFileSettings(\
            directory=problem_parameters.output_settings.directory, \
            number_time_steps=problem_parameters.opf_time_settings.date_time_settings.size, \
            format=problem_parameters.output_settings.format)
        return output_settings
    
    def define_simulation_settings(self, problem_parameters: Parameters):
        if problem_parameters.pandapower_optimisation_settings.initialised and \
            problem_parameters.opf_time_settings.initialised and \
            problem_parameters.problem_settings.initialised:
            self.simulation_settings = SimulationSettings(
                time_steps=range(problem_parameters.opf_time_settings.date_time_settings.size), \
                display_progress_bar = \
                    problem_parameters.pandapower_optimisation_settings.display_progress_bar, \
                continue_on_divergence = \
                    problem_parameters.pandapower_optimisation_settings.continue_on_divergence, \
                optimisation_software = \
                    problem_parameters.pandapower_optimisation_settings.optimisation_software, \
                opf_type = problem_parameters.problem_settings.opf_type)
    
    def run_timestep_opf_pandapower(self):
        self.wrapper.run_timestep_simulation(self.simulation_settings)

    def update_network_controllers(self, pp_profiles_data: PandaPowerProfilesData):
        if pp_profiles_data.initialised:
            for profile in pp_profiles_data.data:
                pp_profile = Profile(\
                    components_indexes_in_power_system=profile.indexes, \
                    data=profile.data, column_names=profile.active_columns_names, \
                    variable_name=profile.variable_name, components_type=profile.element_type)
                self.wrapper.update_network_controller(pp_profile)
