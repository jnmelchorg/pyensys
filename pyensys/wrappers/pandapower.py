from os import name
from pandapower.converter import from_mpc
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower import runopp, runpm_ac_opf
from pandapower.timeseries.run_time_series import run_timeseries

from pandas import DataFrame

from dataclasses import dataclass, field

from typing import List

optimal_power_flow_software_options = {
    'pypower' : runopp,
    'powermodels' : runpm_ac_opf
}

@dataclass
class Profile:
    components_indexes_in_power_system: list = field(default_factory=list)
    data: DataFrame = field(default_factory=DataFrame)
    name: list = field(default_factory=list)
    variable_name: str = ''
    components_type:str = ''

@dataclass
class TimeSeriesOutputFileSettings:
    number_time_steps: int = 0
    directory: str = ''
    format: str = ''

@dataclass
class OutputVariableSet:
    name_dataset: str = ''
    name_variable: str = ''
    variable_indexes : list = field(default_factory=list)
    evaluation_function = None

@dataclass
class SimulationSettings:
    display_progress_bar: bool = False
    optimisation_software: str = ''
    continue_on_divergence: str = False
    time_steps: list = field(default_factory=list)

class PandapowerWrapper:

    nodes_list = []

    def load_mpc_file_to_pandapower(self, filename_with_extension: str,
        frequency_hz: float):
        self.network = from_mpc(filename_with_extension, f_hz=frequency_hz)

    def create_controllers(self, profiles: List[Profile]):
        for profile in profiles:
            self._create_controller(profile)
    
    def _create_controller(self, profile: Profile):
        ConstControl(self.network, element=profile.components_type, 
            element_index=profile.components_indexes_in_power_system,
            variable=profile.variable_name,
            data_source=DFData(profile.data),
            profile_name=profile.name)
    

    def create_output_writer(self, output_properties: TimeSeriesOutputFileSettings, \
        variables_to_save: List[OutputVariableSet]):
        range_timesteps = range(0, output_properties.number_time_steps)
        writer = OutputWriter(self.network, range_timesteps, 
            output_path=output_properties.directory, 
            output_file_type=output_properties.format, log_variables=list())
        self._log_variables_in_output_writer(writer, variables_to_save)
        
    def _log_variables_in_output_writer(self, writer: OutputWriter, 
        variables_to_save: List[OutputVariableSet]):
        for variable in variables_to_save:
            writer = self._log_variable_in_output_writer(writer, variable)
    
    def _log_variable_in_output_writer(self, writer: OutputWriter, 
        variable_set: OutputVariableSet) -> OutputWriter:
        writer.log_variable(variable_set.name_dataset, variable_set.name_variable, 
            index=(variable_set.variable_indexes if variable_set.variable_indexes \
            else None), eval_function=variable_set.evaluation_function)
        return writer


    def run_timestep_simulation(self, settings: SimulationSettings):
        run_timeseries(self.network, time_steps=settings.time_steps,
            continue_on_divergence=settings.continue_on_divergence,
            verbose=settings.display_progress_bar, 
            run=optimal_power_flow_software_options[settings.optimisation_software])
    