from pandapower.converter import from_mpc
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower import pandapowerNet, runopp, runpm_ac_opf
from pandapower.timeseries.run_time_series import run_timeseries

from pandas import DataFrame

from dataclasses import dataclass, field

optimal_power_flow_software_options = {
    'pypower' : runopp,
    'powermodels' : runpm_ac_opf
}

@dataclass
class profile:
    components_indexes_in_power_system: list = field(default_factory=list)
    data: DFData = field(default_factory=DFData)
    name: list = field(default_factory=list)
    variable_name: str = ''
    components_type:str = ''

@dataclass
class time_series_output_file:
    number_time_steps: int = 0
    directory: str = ''
    format: str = ''

@dataclass
class output_variable_set:
    name_dataset: str = ''
    name_variable: str = ''
    variable_indexes : list = field(default_factory=list)
    evaluation_function = None

@dataclass
class simulation_settings:
    display_progress_bar: bool = False
    optimisation_software: str = ''
    continue_on_divergence: str = False
    time_steps: list = field(default_factory=list)


class pandapower_wrapper:
    def load_mpc_file_to_pandapower(self, filename_with_extension: str,
        frequency_hz: float) -> pandapowerNet:
        pandapower_network = from_mpc(filename_with_extension, f_hz=frequency_hz)
        return pandapower_network

    def create_controller(self, network: pandapowerNet, profile: profile) \
        -> ConstControl:
        return ConstControl(network, element=profile.components_type, 
            element_index=profile.components_indexes_in_power_system,
            variable=profile.variable_name,
            data_source=profile.data,
            profile_name=profile.name)

    def create_output_writer(self, network: pandapowerNet, 
        output_properties: time_series_output_file) -> OutputWriter:
        range_timesteps = range(0, output_properties.number_time_steps)
        return OutputWriter(network, range_timesteps, 
            output_path=output_properties.directory, 
            output_file_type=output_properties.format, log_variables=list())
    
    def log_variables_in_output_writer(self, writer: OutputWriter, 
        variable_set: output_variable_set):
        writer.log_variable(variable_set.name_dataset, variable_set.name_variable, 
            index=(variable_set.variable_indexes if variable_set.variable_indexes \
            else None), eval_function=variable_set.evaluation_function)

    def run_timestep_simulation(self, network: pandapowerNet, 
        settings: simulation_settings):
        run_timeseries(network, time_steps=settings.time_steps,
            continue_on_divergence=settings.continue_on_divergence,
            verbose=settings.display_progress_bar, 
            run=optimal_power_flow_software_options[settings.optimisation_software])

    def create_profiles(self, data_set: DataFrame) -> DFData:
        return DFData(data_set)
    