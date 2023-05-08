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
                self._parameters.optimisation_profiles_dataframes.get("flexible units") is not None:
            generators_data = self._parameters.optimisation_profiles_dataframes.get("flexible units")
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
            # print('list_parameter_data...')
            # print(list_parameter_data)
        for parameter in list_parameter_data:
            # print('parameter:')
            # print(parameter)
            if parameter.component_type == "load":
                self.wrapper.network[parameter.component_type].at[
                    list(self.wrapper.network[parameter.component_type][
                        self.wrapper.network[parameter.component_type]["bus"] ==
                        parameter.parameter_position].index)[0],
                    parameter.parameter_name] = parameter.new_value
            else:
                # print('list_parameter_data:')
                # print(list_parameter_data)
                # print(list_parameter_data[parameter].parameter_position)
                # print('list_parameter_data[parameter].parameter_position')
                print('\nprint parameter.parameter_position...')
                print(parameter.parameter_position)
                # print('new_line_parameter_count:')
                # print(new_line_parameter_count)
                # if len(parameter.parameter_position) == 1:
                if isinstance(parameter.parameter_position, int) == True: # if investment option is just a single line (this was added by Wangwei and Andrey during the tests)
                    # self.wrapper.network[parameter.component_type].at[ \
                    #     parameter.parameter_position, parameter.parameter_name] = parameter.new_value # activating new lines - modify this?

                    self.wrapper.network[parameter.component_type].at[ \
                        parameter.parameter_position, 'in_service'] = True # - modified - making lines active anyway

                    volt_bus = self.wrapper.network.bus["vn_kv"][self.wrapper.network.line.from_bus[parameter.parameter_position]]
                    # print('volt_bus: ',volt_bus)

                    line_impedance_factor1 = self.wrapper.network[parameter.component_type].at[parameter.parameter_position, 'max_i_ka']

                    self.wrapper.network[parameter.component_type].at[ \
                        parameter.parameter_position, 'max_i_ka'] += parameter.capacity_to_be_added_MW*1e6/(volt_bus*1e3)/(3**0.5)/1e3 # calculating the line limit in kA
                    print('\nparameter.capacity_to_be_added_MW:')
                    print(parameter.capacity_to_be_added_MW)
                    
                    line_impedance_factor2 = self.wrapper.network[parameter.component_type].at[parameter.parameter_position, 'max_i_ka']

                    line_impedance_factor = line_impedance_factor1/line_impedance_factor2 # decreasing impedance of updated lines
                    print('line_impedance_factor: ',line_impedance_factor)
                    
                    self.wrapper.network[parameter.component_type].at[ \
                    parameter.parameter_position, 'r_ohm_per_km'] *= line_impedance_factor
                    self.wrapper.network[parameter.component_type].at[ \
                    parameter.parameter_position, 'x_ohm_per_km'] *= line_impedance_factor

                    # print("parameter.parameter_position: ",parameter.parameter_position)
                    # print('self._parameters.optimisation_binary_variables:')
                    # print(self._parameters.optimisation_binary_variables[0].capacity_to_be_added_MW[new_line_parameter_count])
                    # print('parameter.parameter_position')
                    # print(parameter.parameter_position)

                    # self.wrapper.network.line.max_i_ka[parameter.parameter_position] = 555 # for testing purposes

                    # self.wrapper.network.line.max_i_ka[parameter.parameter_position] += \
                    # self._parameters.optimisation_binary_variables[0].capacity_to_be_added_MW[0]*1e6/(volt_bus*1e3)/(3**0.5)/1e3 # calculating the line limit in kA
                    
                    # print("parameter.parameter_position: ",parameter.parameter_position)
                    # print("self.wrapper.network.line: ")
                    # print(self.wrapper.network.line)

                    # print('self._parameters.optimisation_binary_variables:')
                    # print(self._parameters.optimisation_binary_variables[0].capacity_to_be_added_MW)

                else: # if investment option is a cluster of lines (this was added by Wangwei and Andrey during the tests)
                    for iii in range(len(parameter.parameter_position)): 
                        print()
                        self.wrapper.network['line'].at[ \
                        parameter.parameter_position[iii]-1, 'in_service'] = True # activating new lines (old approach)

                        volt_bus = self.wrapper.network.bus["vn_kv"][self.wrapper.network.line.from_bus[parameter.parameter_position[iii]-1]]
                        # print('volt_bus:')
                        # print(volt_bus)

                        line_impedance_factor1 = self.wrapper.network[parameter.component_type].at[parameter.parameter_position[iii]-1, 'max_i_ka']
                        # print('max_i_ka initial:')
                        # print(self.wrapper.network[parameter.component_type].at[parameter.parameter_position[iii]-1, 'max_i_ka'])
                        self.wrapper.network[parameter.component_type].at[ \
                        parameter.parameter_position[iii]-1, 'max_i_ka'] += parameter.capacity_to_be_added_MW[iii]*1e6/(volt_bus*1e3)/(3**0.5)/1e3 # calculating the new line limit in kA
                        # print('max_i_ka to be added:')
                        # print(parameter.capacity_to_be_added_MW[iii]*1e6/(volt_bus*1e3)/(3**0.5)/1e3)
                        # print('max_i_ka new:')
                        # print(self.wrapper.network[parameter.component_type].at[parameter.parameter_position[iii]-1, 'max_i_ka'])

                        # self.wrapper.network[parameter.component_type].at[ \
                        # parameter.parameter_position[iii]-1, 'max_i_ka'] += 555 # adding large capacity for testing
                                                
                        print('Increasing capacity of line ',parameter.parameter_position[iii]-1,'by ',parameter.capacity_to_be_added_MW[iii],' MW')
                        print('parameter.capacity_to_be_added_MW: ', parameter.capacity_to_be_added_MW)

                        line_impedance_factor2 = self.wrapper.network[parameter.component_type].at[parameter.parameter_position[iii]-1, 'max_i_ka']

                        line_impedance_factor = line_impedance_factor1/line_impedance_factor2 # decreasing impedance of updated lines
                        # print('line_impedance_factor: ',line_impedance_factor)
                        
                        self.wrapper.network[parameter.component_type].at[ \
                        parameter.parameter_position[iii]-1, 'r_ohm_per_km'] *= line_impedance_factor
                        self.wrapper.network[parameter.component_type].at[ \
                        parameter.parameter_position[iii]-1, 'x_ohm_per_km'] *= line_impedance_factor


                        print("parameter.parameter_position: ",parameter.parameter_position)
                        # print("self.wrapper.network.line: ")
                        # print(self.wrapper.network.line)

                        # print('self._parameters.optimisation_binary_variables:')
                        # print(self._parameters.optimisation_binary_variables[0].capacity_to_be_added_MW)
                        print()
                        


    def get_total_cost(self) -> float:
        return self.wrapper.get_total_cost()

    def run_ac_opf_pandapower(self):
        self.wrapper.run_ac_opf(self.simulation_settings)

    def _get_index_of_element_based_on_parameter(self, element_name: str, parameter_name: str, value):
        return self.wrapper.network[element_name][self.wrapper.network[element_name][parameter_name] == value].index[0]

    def _get_index_of_element_based_on_multiple_parameters(self, element_name: str, parameters_names: List[str],
                                                           values: list):
        if len(parameters_names) != len(values):
            raise ValueError("The number of parameters and values must be equal")
        row = self.wrapper.network[element_name]
        for parameter_name, value in zip(parameters_names, values):
            row = row[row[parameter_name] == value]
        return row.index[0]
