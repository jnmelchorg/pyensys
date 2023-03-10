from typing import List

from pandapower.converter import from_mpc
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower import runopp, runpm_ac_opf, create_empty_network, create_sgen, create_poly_cost, create_ext_grid
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower import runpp

from pyensys.wrappers.PandapowerDataClasses import OutputVariableSet, Profile, TimeSeriesOutputFileSettings, \
    SimulationSettings

OPTIMAL_POWER_FLOW_SOFTWARE_OPTIONS = {
    "ac": {
        'pypower': runopp, # runopp
        'power models': runpm_ac_opf
    }
}


def log_variable_in_output_writer(writer: OutputWriter,
                                  variable_set: OutputVariableSet) -> OutputWriter:
    writer.log_variable(variable_set.name_dataset, variable_set.name_variable,
                        index=(variable_set.variable_indexes if variable_set.variable_indexes
                               else None), eval_function=variable_set.evaluation_function)
    return writer


def log_variables_in_output_writer(writer: OutputWriter,
                                   variables_to_save: List[OutputVariableSet]) -> OutputWriter:
    for variable in variables_to_save:
        writer = log_variable_in_output_writer(writer, variable)
    return writer


class PandaPowerWrapper:
    def __init__(self):
        self.network = create_empty_network()

    def load_mat_file_to_pandapower(self, file_path_with_extension: str,
                                    frequency_hz: float, case_name_in_mpc_file: str = "mpc"):
        self.network = from_mpc(file_path_with_extension, f_hz=frequency_hz, casename_mpc_file=case_name_in_mpc_file)
        # self.network.ext_grid.at[0,"vm_pu"] = 1.06 # volage tests
        # bus = self.network.ext_grid.at[0,"vm_pu"] = 1.038 # volage tests

    def add_controllers_to_network(self, profiles: List[Profile]):
        for profile in profiles:
            self.add_controller_to_network(profile)

    def add_controller_to_network(self, profile: Profile):
        ConstControl(self.network, element=profile.components_type,
                     element_index=profile.components_indexes_in_power_system,
                     variable=profile.variable_name,
                     data_source=DFData(profile.data),
                     profile_name=profile.column_names)

    def add_output_writer_to_network(self, output_properties: TimeSeriesOutputFileSettings,
                                     variables_to_save: List[OutputVariableSet]):
        if output_properties.number_time_steps is not None:
            range_time_steps = range(0, output_properties.number_time_steps)
        else:
            range_time_steps = None
        writer = OutputWriter(self.network, range_time_steps,
                              output_path=output_properties.directory,
                              output_file_type=output_properties.format, log_variables=list())
        writer = log_variables_in_output_writer(writer, variables_to_save)

    def run_time_step_simulation(self, settings: SimulationSettings):
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

    def is_feasible(self) -> bool:
        return self.network.OPF_converged

    def create_static_generator(self, bus_index) -> int:
        return create_sgen(self.network, bus=bus_index, p_mw=0.0, controllable=True, min_p_mw=0.0, max_p_mw=0.0,
                           min_q_mvar=0.0, max_q_mvar=0.0)

    def create_poly_cost(self, index, type_element):
        create_poly_cost(self.network, element=index, et=type_element, cp1_eur_per_mw=0.0)

    def get_total_cost(self) -> float:
        return self.network.res_cost

    def run_ac_opf(self, settings: SimulationSettings):
        try:
            if settings.optimisation_software == "pypower":
                OPTIMAL_POWER_FLOW_SOFTWARE_OPTIONS[settings.opf_type][
                    # settings.optimisation_software](self.network, verbose=settings.display_progress_bar, numba=False)
                    settings.optimisation_software](self.network, verbose=0, numba=False) # change verbose=0 to silence the solver
            elif settings.optimisation_software == "power models":
                OPTIMAL_POWER_FLOW_SOFTWARE_OPTIONS[settings.opf_type][
                    settings.optimisation_software](self.network, silence=not settings.display_progress_bar)
        except:
            pass
