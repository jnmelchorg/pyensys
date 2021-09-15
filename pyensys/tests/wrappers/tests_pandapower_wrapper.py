from pyensys.wrappers.pandapower import PandapowerWrapper, Profile, \
    TimeSeriesOutputFileSettings, OutputVariableSet, SimulationSettings

from pyensys.tests.tests_data_paths import get_path_case9_mat

from pandas import DataFrame
from numpy.random import random

from pandapower.timeseries.output_writer import OutputWriter
from pandapower import runopp

from typing import List
from math import isclose

HOURS_IN_DAY: int = 24
DUMMY_POWER_GENERATION: float = 250
DUMMY_POWER_DEMAND: float = 90

def load_test_network() -> PandapowerWrapper:
    wrapper = PandapowerWrapper()
    file = get_path_case9_mat()
    wrapper.load_mpc_file_to_pandapower(file, 60.0)
    return wrapper

def create_dummy_profile_data() -> DataFrame:
    profiles = DataFrame()
    profiles['load1_p'] = [67.28095505, 9.65466896, 11.70181664, \
        3.43219975, 32.61755449, 84.85360346, 29.95258074, \
        39.20088948, 42.54484985, 20.04279764, 71.8828131, \
        41.96986409, 7.21552544, 25.88618743, 26.54954622, \
        17.21125963, 87.53580282, 87.63197474, 69.49124619, \
        1.3286569, 10.60700364, 54.86348133, 13.76146653, 18.54152069]
    profiles['gen1_p'] = [240.44092015, 205.50525905, 18.7321705, \
        136.89180324, 227.91854659, 243.98374614, 20.22587417, \
        204.13000456, 142.52170518, 69.15390821, 75.08460741, \
        100.18795258, 177.58383221, 105.98171684, 111.27365188, \
        177.83779845, 163.92025682, 148.67520413, \
        28.95077739, 116.63621831, 126.36468424, 164.58739745, \
        71.51992487, 77.18482911]
    return profiles

def create_dummy_profile_object(data: DataFrame, indexes: list, 
    profile_name: list, variable_name: str, element_type: str) -> Profile:
    prof = Profile(data=data.loc[:, profile_name[0]].to_frame())
    prof.components_indexes_in_power_system = indexes
    prof.name = profile_name
    prof.variable_name = variable_name
    prof.components_type = element_type
    return prof

def create_dummy_profiles() -> List[Profile]:
    profiles = []
    data = create_dummy_profile_data()
    profiles.append(create_dummy_profile_object(data, [0], ["load1_p"],\
        'p_mw', 'load'))
    profiles.append(create_dummy_profile_object(data, [0], ["gen1_p"],\
        'p_mw', 'gen'))
    return profiles

def create_dummy_output_file_settings() -> TimeSeriesOutputFileSettings:
    output_settings = TimeSeriesOutputFileSettings()
    output_settings.directory = \
       'C:\\Users\\f09903jm\\git projects\\pyensys\\tests\\outputs'
    output_settings.number_time_steps = HOURS_IN_DAY
    output_settings.format = '.xlsx'
    return output_settings

def create_pandapower_writer(wrapper: PandapowerWrapper) -> PandapowerWrapper:
    output_settings = create_dummy_output_file_settings()
    variables_to_save = create_dummy_output_variables()
    wrapper.create_output_writer(output_settings, variables_to_save)
    return wrapper

def create_dummy_output_variables() -> List[OutputVariableSet]:
    variables = []
    variables.append(OutputVariableSet('res_load', 'p_mw', []))
    variables.append(OutputVariableSet('res_bus', 'vm_pu', []))
    variables.append(OutputVariableSet('res_line', 'loading_percent', []))
    variables.append(OutputVariableSet('res_line', 'i_ka', []))
    return variables

def create_dummy_controllers(wrapper: PandapowerWrapper) -> PandapowerWrapper:
    profiles = create_dummy_profiles()
    wrapper.create_controllers(profiles)
    return wrapper

def create_dummy_network_with_settings() -> PandapowerWrapper:
    wrapper = load_test_network()
    wrapper = create_dummy_controllers(wrapper)
    wrapper = create_pandapower_writer(wrapper)
    return wrapper

def create_dummy_simulation_settings() -> SimulationSettings:
    settings = SimulationSettings()
    settings.time_steps = range(HOURS_IN_DAY)
    settings.display_progress_bar = False
    settings.continue_on_divergence = False
    settings.optimisation_software = 'pypower'
    return settings


def test_if_load_mpc_file_to_pandapower_works():
    wrapper = PandapowerWrapper()
    file = get_path_case9_mat()
    wrapper.load_mpc_file_to_pandapower(file, 50.0)
    assert len(wrapper.network['bus'].index) == 9

def test_create_controller():
    wrapper = load_test_network()
    profiles = create_dummy_profiles()
    wrapper._create_controller(profiles[0])
    data = wrapper.network['controller'].iat[0, 0].data_source.df
    profile_name = wrapper.network['controller'].iat[0, 0].profile_name
    assert len(data.index) == 24
    assert profile_name == ['load1_p']

def test_create_controllers():
    wrapper = load_test_network()
    profiles = create_dummy_profiles()
    wrapper.create_controllers(profiles)
    data_demand_profile = wrapper.network['controller'].iat[0, 0].data_source.df
    data_generation_profile = wrapper.network['controller'].iat[1, 0].data_source.df
    profile_load_name = wrapper.network['controller'].iat[0, 0].profile_name
    profile_generation_name = wrapper.network['controller'].iat[1, 0].profile_name
    assert len(wrapper.network['controller'].index) == 2
    assert profile_load_name == ['load1_p'] and profile_generation_name == ['gen1_p']
    assert len(data_demand_profile.index) == 24 and len(data_generation_profile.index) == 24

def test_log_variable_in_writer():
    wrapper = load_test_network()
    writer = OutputWriter(wrapper.network)
    variable_set = OutputVariableSet('res_bus', 'p_mw', [])
    wrapper._log_variable_in_output_writer(writer, variable_set)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert output_writer_class.log_variables[2] == ('res_bus', 'p_mw', None, None, None)

def test_log_variables_in_writer():
    wrapper = load_test_network()
    writer = OutputWriter(wrapper.network)
    variables_to_save = [OutputVariableSet('res_bus', 'p_mw', []), \
        OutputVariableSet('res_line', 'i_ka', [])]
    wrapper._log_variables_in_output_writer(writer, variables_to_save)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert output_writer_class.log_variables[2] == ('res_bus', 'p_mw', None, None, None) and \
        output_writer_class.log_variables[3] == ('res_line', 'i_ka', None, None, None)

def test_output_writer():
    wrapper = load_test_network()
    wrapper = create_pandapower_writer(wrapper)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert output_writer_class.log_variables[2] == \
        ('res_line', 'loading_percent', None, None, None)

def test_run_timestep_simulation():
    wrapper = create_dummy_network_with_settings()
    settings = create_dummy_simulation_settings()
    wrapper.run_timestep_simulation(settings)
    assert wrapper.network.OPF_converged == True
    assert isclose(3714.91365, wrapper.network.res_cost, abs_tol=1e-4)