from pyensys.wrappers.PandaPowerWrapper import PandaPowerWrapper
from pyensys.tests.tests_data_paths import get_path_case9_mat, set_pandapower_test_output_directory
from pandas import DataFrame
from pandapower.timeseries.output_writer import OutputWriter
from math import isclose
from pyensys.wrappers.PandapowerDataClasses import *


def test_if_load_mat_file_to_pandapower_works():
    wrapper = PandaPowerWrapper()
    file = get_path_case9_mat()
    wrapper.load_mat_file_to_pandapower(file, 50.0)
    assert len(wrapper.network['bus'].index) == 9

def test_add_controller_to_network():
    wrapper = PandaPowerWrapper()
    data = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile = Profile(data=data, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    wrapper.add_controller_to_network(profile)
    data = wrapper.network['controller'].iat[0, 0].data_source.df
    profile_name = wrapper.network['controller'].iat[0, 0].profile_name
    assert len(data.index) == 3
    assert profile_name == ['load1_p']

def test_create_controllers():
    wrapper = PandaPowerWrapper()
    data1 = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], columns=['gen1_p'])
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    data_demand_profile = wrapper.network['controller'].iat[0, 0].data_source.df
    data_generation_profile = wrapper.network['controller'].iat[1, 0].data_source.df
    profile_load_name = wrapper.network['controller'].iat[0, 0].profile_name
    profile_generation_name = wrapper.network['controller'].iat[1, 0].profile_name
    assert len(wrapper.network['controller'].index) == 2
    assert profile_load_name == ['load1_p'] and profile_generation_name == ['gen1_p']
    assert len(data_demand_profile.index) == 3 and len(data_generation_profile.index) == 3

def test_log_variable_in_writer():
    wrapper = PandaPowerWrapper()
    writer = OutputWriter(wrapper.network)
    variable_set = OutputVariableSet('res_bus', 'p_mw', [])
    writer = wrapper.log_variable_in_output_writer(writer, variable_set)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert output_writer_class.log_variables[2] == ('res_bus', 'p_mw', None, None, None)

def test_log_variables_in_writer():
    wrapper = PandaPowerWrapper()
    writer = OutputWriter(wrapper.network)
    variables_to_save = [OutputVariableSet('res_bus', 'p_mw', []), \
        OutputVariableSet('res_line', 'i_ka', [])]
    writer = wrapper.log_variables_in_output_writer(writer, variables_to_save)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert output_writer_class.log_variables[2] == ('res_bus', 'p_mw', None, None, None) and \
        output_writer_class.log_variables[3] == ('res_line', 'i_ka', None, None, None)

def test_output_writer():
    wrapper = PandaPowerWrapper()
    variables = [OutputVariableSet('res_load', 'p_mw', []), OutputVariableSet('res_bus', 'vm_pu', []), \
        OutputVariableSet('res_line', 'loading_percent', []), OutputVariableSet('res_line', 'i_ka', [])]
    output_settings = TimeSeriesOutputFileSettings(directory=set_pandapower_test_output_directory(), \
        number_time_steps=3, format='.xlsx')
    wrapper.add_output_writer_to_network(output_settings, variables)
    output_writer_class = wrapper.network['output_writer'].iat[0, 0]
    assert len(output_writer_class.log_variables) == 4
    assert output_writer_class.log_variables[2] == ('res_line', 'loading_percent', None, None, None)

def test_run_timestep_simulation():
    wrapper = PandaPowerWrapper()
    wrapper.load_mat_file_to_pandapower(filename_with_extension=get_path_case9_mat(), frequency_hz=60.0)
    data1 = DataFrame()
    data1['load1_p'] = [67.28095505, 9.65466896, 11.70181664]
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame()
    data2['gen1_p'] = [240.44092015, 205.50525905, 18.7321705]
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    variables = [OutputVariableSet('res_load', 'p_mw', []), OutputVariableSet('res_bus', 'vm_pu', []), \
        OutputVariableSet('res_line', 'loading_percent', []), OutputVariableSet('res_line', 'i_ka', [])]
    output_settings = TimeSeriesOutputFileSettings(directory=set_pandapower_test_output_directory(), \
        number_time_steps=3, format='.xlsx')
    wrapper.add_output_writer_to_network(output_settings, variables)
    settings = SimulationSettings(time_steps=range(3), display_progress_bar = False, \
        continue_on_divergence = False, optimisation_software = 'pypower', opf_type = 'ac')
    wrapper.run_timestep_simulation(settings)
    assert wrapper.network.OPF_converged == True
    assert isclose(3583.53647, wrapper.network.res_cost, abs_tol=1e-4)

def test_get_row_to_update_case1():
    wrapper = PandaPowerWrapper()
    data1 = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], columns=['gen1_p'])
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    assert wrapper.get_row_to_update(profile2) == 1

def test_get_row_to_update_case2():
    wrapper = PandaPowerWrapper()
    data1 = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], columns=['gen1_p'])
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    profile2.variable_name = "x"
    assert wrapper.get_row_to_update(profile2) == -1

def test_get_row_to_drop_case3():
    wrapper = PandaPowerWrapper()
    data1 = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], columns=['gen1_p'])
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    profile2.components_type = "x"
    assert wrapper.get_row_to_update(profile2) == -1

def test_update_controller_data():
    wrapper = PandaPowerWrapper()
    data1 = DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], columns=['load1_p'])
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], columns=['gen1_p'])
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    data3 = DataFrame(data=[[220], [190], [5]], columns=['gen1_p'])
    profile3 = Profile(data=data3, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    wrapper.update_network_controller(profile3)
    assert data3.equals(wrapper.network['controller'].iat[1, 0].data_source.df)

def test_is_feasible():
    wrapper = PandaPowerWrapper()
    wrapper.load_mat_file_to_pandapower(filename_with_extension=get_path_case9_mat(), frequency_hz=60.0)
    data1 = DataFrame()
    data1['load1_p'] = [67.28095505, 9.65466896, 11.70181664]
    profile1 = Profile(data=data1, components_indexes_in_power_system=[0], \
        column_names=['load1_p'], variable_name= "p_mw", components_type="load")
    data2 = DataFrame()
    data2['gen1_p'] = [240.44092015, 205.50525905, 18.7321705]
    profile2 = Profile(data=data2, components_indexes_in_power_system=[0], \
        column_names=["gen1_p"], variable_name= "p_mw", components_type="gen")
    profiles = [profile1, profile2]
    wrapper.add_controllers_to_network(profiles=profiles)
    variables = [OutputVariableSet('res_load', 'p_mw', []), OutputVariableSet('res_bus', 'vm_pu', []), \
        OutputVariableSet('res_line', 'loading_percent', []), OutputVariableSet('res_line', 'i_ka', [])]
    output_settings = TimeSeriesOutputFileSettings(directory=set_pandapower_test_output_directory(), \
        number_time_steps=3, format='.xlsx')
    wrapper.add_output_writer_to_network(output_settings, variables)
    settings = SimulationSettings(time_steps=range(3), display_progress_bar = False, \
        continue_on_divergence = False, optimisation_software = 'pypower', opf_type = 'ac')
    wrapper.run_timestep_simulation(settings)
    assert wrapper.is_feasible() == True
