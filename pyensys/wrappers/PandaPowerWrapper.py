from pandapower.converter import from_mpc
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower import runopp, runpm_ac_opf, create_empty_network
from pandapower.timeseries.run_time_series import run_timeseries
from typing import List
from pyensys.wrappers.PandapowerDataClasses import *

OPTIMAL_POWER_FLOW_SOFTWARE_OPTIONS = {
    "ac": {
        'pypower' : runopp,
        'powermodels' : runpm_ac_opf
    }
}

class PandaPowerWrapper:
    def __init__(self):
        self.network = create_empty_network()

    def load_mat_file_to_pandapower(self, filename_with_extension: str,
        frequency_hz: float):
        self.network = from_mpc(filename_with_extension, f_hz=frequency_hz)

    def add_controllers_to_network(self, profiles: List[Profile]):
        for profile in profiles:
            self.add_controller_to_network(profile)

    def add_controller_to_network(self, profile: Profile):
        ConstControl(self.network, element=profile.components_type, 
            element_index=profile.components_indexes_in_power_system,
            variable=profile.variable_name,
            data_source=DFData(profile.data),
            profile_name=profile.column_names)

    def add_output_writer_to_network(self, output_properties: TimeSeriesOutputFileSettings, \
        variables_to_save: List[OutputVariableSet]):
        range_timesteps = range(0, output_properties.number_time_steps)
        writer = OutputWriter(self.network, range_timesteps, 
            output_path=output_properties.directory, 
            output_file_type=output_properties.format, log_variables=list())
        writer = self.log_variables_in_output_writer(writer, variables_to_save)

    def log_variables_in_output_writer(self, writer: OutputWriter, \
        variables_to_save: List[OutputVariableSet]) -> OutputWriter:
        for variable in variables_to_save:
            writer = self.log_variable_in_output_writer(writer, variable)
        return writer

    def log_variable_in_output_writer(self, writer: OutputWriter, 
        variable_set: OutputVariableSet) -> OutputWriter:
        writer.log_variable(variable_set.name_dataset, variable_set.name_variable, 
            index=(variable_set.variable_indexes if variable_set.variable_indexes \
            else None), eval_function=variable_set.evaluation_function)
        return writer

    def run_timestep_simulation(self, settings: SimulationSettings):
        available_software_per_type = OPTIMAL_POWER_FLOW_SOFTWARE_OPTIONS[settings.opf_type]
        run_timeseries(self.network, time_steps=settings.time_steps,
            continue_on_divergence=settings.continue_on_divergence,
            verbose=settings.display_progress_bar, 
            run=available_software_per_type[settings.optimisation_software])

    def update_network_controller(self, new_profile: Profile):
        row_to_update = self.get_row_to_update(new_profile)
        self.network['controller'].iat[row_to_update, 0].data_source.df = new_profile.data

    def get_row_to_update(self, new_profile: Profile) -> int:
        row_counter = 0
        for row in self.network['controller'].itertuples():
            element = self.network['controller'].iat[row_counter, 0].element
            variable = self.network['controller'].iat[row_counter, 0].variable
            row_counter += 1
            if element == new_profile.components_type and variable == new_profile.variable_name:
                return row[0]
        return -1
