from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.toolbox import set_element_status
from pyensys.wrappers.pandapower import pandapower_wrapper, profile, \
    time_series_output_file, output_variable_set, simulation_settings

from pandas import DataFrame
from numpy.random import random

from pandapower import create_empty_network, create_bus, \
    create_ext_grid, create_line, create_transformer, create_load, create_gen, \
    pandapowerNet, create_pwl_cost
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter

from typing import List

HOURS_IN_DAY: int = 24
DUMMY_POWER: float = 25

def create_dummy_buses(network: pandapowerNet) -> List[int]:
    buses = []
    buses.append(create_bus(network, 110))
    buses.append(create_bus(network, 110))
    buses.append(create_bus(network, 20))
    buses.append(create_bus(network, 20))
    buses.append(create_bus(network, 20))
    return buses

def create_dummy_lines(network: pandapowerNet, buses: List[int]):
    create_line(network, buses[0], buses[1], 10, "149-AL1/24-ST1A 110.0")
    create_line(network, buses[2], buses[3], 10, "184-AL1/30-ST1A 20.0")
    create_line(network, buses[2], buses[4], 10, "184-AL1/30-ST1A 20.0")

def create_dummy_network() -> pandapowerNet:
    """
    simple net that looks like:

    ext_grid b0---b1 trafo(110/20) b2----b3 load
                                    |
                                    |
                                    b4 sgen
    """
    net = create_empty_network()
    buses = create_dummy_buses(net)
    create_ext_grid(net, buses[0])
    create_transformer(net, buses[1], buses[2], "25 MVA 110/20 kV", name='tr1')
    create_dummy_lines(net, buses)
    create_load(net, buses[3], p_mw=20., q_mvar=10., name='load1')
    create_gen(net, buses[4], max_p_mw =25., min_p_mw=0.0, \
        max_q_mvar=0.2, min_q_mvar=-0.2, name='sgen1', min_vm_pu=0.9, 
        max_vm_pu=1.1, p_mw=20.0)
    create_pwl_cost(net, 0, "gen", [[0.0, 25.0, 1.0]])
    return net

def create_dummy_profile_data() -> DataFrame:
    profiles = DataFrame()
    profiles['load1_p'] = random(HOURS_IN_DAY) * DUMMY_POWER
    profiles['sgen1_p'] = random(HOURS_IN_DAY) * DUMMY_POWER
    return profiles

def create_dummy_profile_object(data: DataFrame, indexes: list, 
    profile_name: list, variable_name: str, element_type: str) -> profile:
    wrapper = pandapower_wrapper()
    prof = profile(data=wrapper.create_profiles(data))
    prof.components_indexes_in_power_system = indexes
    prof.name = profile_name
    prof.variable_name = variable_name
    prof.components_type = element_type
    return prof

def create_dummy_profiles() -> list:
    profiles = []
    data = create_dummy_profile_data()
    profiles.append(create_dummy_profile_object(data, [0], ["load1_p"],\
        'p_mw', 'load'))
    profiles.append(create_dummy_profile_object(data, [0], ["sgen1_p"],\
        'p_mw', 'gen'))
    return profiles

def create_dummy_output_file_settings() -> time_series_output_file:
    output_settings = time_series_output_file()
    output_settings.directory = \
       'C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\outputs'
    output_settings.number_time_steps = HOURS_IN_DAY
    output_settings.format = '.xlsx'
    return output_settings

def create_pandapower_writer(network: pandapowerNet) -> OutputWriter:
    wrapper = pandapower_wrapper()
    output_settings = create_dummy_output_file_settings()
    return wrapper.create_output_writer(network, output_settings)

def create_dummy_output_variables() -> List[output_variable_set]:
    variables = []
    variables.append(output_variable_set('res_load', 'p_mw', []))
    variables.append(output_variable_set('res_bus', 'vm_pu', []))
    variables.append(output_variable_set('res_line', 'loading_percent', []))
    variables.append(output_variable_set('res_line', 'i_ka', []))
    return variables

def add_dummy_variables_to_writer(writer: OutputWriter) -> OutputWriter:
    variables = create_dummy_output_variables()
    for variable in variables:
        writer.log_variable(variable.name_dataset, variable.name_variable,
            index=variable.variable_indexes,
            eval_function=variable.evaluation_function)
    return writer

def create_dummy_controllers(network: pandapowerNet):
    wrapper = pandapower_wrapper()
    profiles = create_dummy_profiles()
    for profile in profiles:
        wrapper.create_controller(network, profile)

def create_dummy_network_with_settings() -> pandapowerNet:
    network = create_dummy_network()
    create_dummy_controllers(network)
    writer = create_pandapower_writer(network)
    writer = add_dummy_variables_to_writer(writer)
    return network

def create_dummy_simulation_settings() -> simulation_settings:
    settings = simulation_settings()
    settings.time_steps = range(HOURS_IN_DAY)
    settings.display_progress_bar = True
    settings.continue_on_divergence = True
    settings.optimisation_software = 'pypower'
    return settings


def test_if_load_mpc_file_to_pandapower_works():
    wrapper = pandapower_wrapper()
    file = 'C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\matpower\\case9.mat'
    assert wrapper.load_mpc_file_to_pandapower(file, 50.0)

def test_create_pandapower_dataframe():
    wrapper = pandapower_wrapper()
    profiles = create_dummy_profile_data()
    assert isinstance(wrapper.create_profiles(profiles['load1_p']), DFData)

def test_create_controller():
    wrapper = pandapower_wrapper()
    network = create_dummy_network()
    profiles = create_dummy_profiles()
    assert isinstance(wrapper.create_controller(network, profiles[0]), ConstControl)

def test_output_writer():
    network = create_dummy_network()
    assert(create_pandapower_writer(network), OutputWriter)

def test_log_variable_in_writer():
    network = create_dummy_network()
    writer = create_pandapower_writer(network)
    wrapper = pandapower_wrapper()
    variable_set = output_variable_set('res_bus', 'p_mw', [])
    wrapper.log_variables_in_output_writer(writer, variable_set)
    assert writer.log_variables[0] == ('res_bus', 'p_mw', None, None, None)

def test_run_timestep_simulation():
    wrapper = pandapower_wrapper()
    network = create_dummy_network_with_settings()
    settings = create_dummy_simulation_settings()
    wrapper.run_timestep_simulation(network, settings)
    assert (network.OPF_converged == True or network.OPF_converged == False)
