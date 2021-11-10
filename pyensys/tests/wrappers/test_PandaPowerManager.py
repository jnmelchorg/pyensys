from numpy.testing._private.utils import assert_equal
from pyensys.wrappers.PandaPowerManager import PandaPowerManager
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable, \
    PandaPowerProfilesData
from pyensys.tests.test_data_paths import get_path_case9_mat, set_pandapower_test_output_directory

from pandas import DataFrame, date_range
from math import isclose
from unittest.mock import MagicMock

def test_load_mat_file_to_pandapower_case1():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_mpc_settings.mat_file_path = get_path_case9_mat()
    parameters.pandapower_mpc_settings.system_frequency = 50.0
    parameters.pandapower_mpc_settings.initialised =  True
    manager._parameters = parameters
    manager.load_mat_file_to_pandapower()
    assert len(manager.wrapper.network['bus'].index) == 9

def test_load_mat_file_to_pandapower_case2():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_mpc_settings.initialised =  False
    manager._parameters = parameters
    manager.load_mat_file_to_pandapower()
    assert len(manager.wrapper.network['bus'].index) == 0

def test_add_controllers_to_network_case1():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
        indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
        columns=['load1_p']), active_columns_names=['load1_p'])]
    manager._parameters = parameters
    manager.add_controllers_to_network()
    data = manager.wrapper.network['controller'].iat[0, 0].data_source.df
    profile_name = manager.wrapper.network['controller'].iat[0, 0].profile_name
    assert len(data.index) == 3
    assert profile_name == ['load1_p']

def test_add_controllers_to_network_case2():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_profiles_data.initialised = False
    manager._parameters = parameters
    manager.add_controllers_to_network()
    assert manager.wrapper.network['controller'].empty

def test_define_output_variables():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.output_variables = [OutputVariable(name_dataset='res_bus', \
        name_variable='p_mw', variable_indexes=[]), OutputVariable(name_dataset='res_line', \
        name_variable='i_ka', variable_indexes=[])]
    manager._parameters = parameters
    variables = manager._define_output_variables()
    assert len(variables) == 2
    assert variables[0].name_dataset == 'res_bus'
    assert variables[1].name_variable == 'i_ka'

def test_define_output_settings():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.directory = set_pandapower_test_output_directory()
    parameters.output_settings.format = ".xlsx"
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    settings = manager._define_output_settings()
    assert settings.directory == set_pandapower_test_output_directory()
    assert settings.format == ".xlsx"
    assert settings.number_time_steps == 3

def test_add_output_writer_to_network_case1():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.initialised = True
    parameters.output_settings.output_variables = [OutputVariable(name_dataset='res_bus', \
        name_variable='p_mw', variable_indexes=[]), OutputVariable(name_dataset='res_line', \
        name_variable='i_ka', variable_indexes=[])]
    parameters.output_settings.directory = set_pandapower_test_output_directory()
    parameters.output_settings.format = ".xlsx"
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.add_output_writer_to_network()
    output_writer_class = manager.wrapper.network['output_writer'].iat[0, 0]
    assert len(output_writer_class.log_variables) == 2
    assert output_writer_class.log_variables[1] == ('res_line', 'i_ka', None, None, None)

def test_add_output_writer_to_network_case2():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.initialised = False
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.add_output_writer_to_network()
    assert not manager.wrapper.network.get('output_writer', False)

def test_add_output_writer_to_network_case3():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.initialised = True
    parameters.output_settings.output_variables = [OutputVariable(name_dataset='res_bus', \
        name_variable='p_mw', variable_indexes=[]), OutputVariable(name_dataset='res_line', \
        name_variable='i_ka', variable_indexes=[])]
    parameters.output_settings.directory = set_pandapower_test_output_directory()
    parameters.output_settings.format = ".xlsx"
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.add_output_writer_to_network()
    assert not manager.wrapper.network.get('output_writer', False)

def test_add_output_writer_to_network_case4():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.output_settings.initialised = False
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.add_output_writer_to_network()
    assert not manager.wrapper.network.get('output_writer', False)

def test_define_simulation_settings_case1():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 3
    assert manager.simulation_settings.opf_type == "ac"
    assert manager.simulation_settings.optimisation_software == "pypower"

def test_define_simulation_settings_case2():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = False
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case3():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = False
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case4():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case5():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = False
    parameters.problem_settings.initialised = False
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case6():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = False
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case7():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = False
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_define_simulation_settings_case8():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_optimisation_settings.initialised = False
    parameters.problem_settings.initialised = False
    parameters.opf_time_settings.initialised = False
    manager._parameters = parameters
    manager.define_simulation_settings()
    assert len(manager.simulation_settings.time_steps) == 0
    assert manager.simulation_settings.opf_type == ""
    assert manager.simulation_settings.optimisation_software == ""

def test_initialise_pandapower_network():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.problem_settings.opf_optimizer = "TEST"
    manager._initialise = MagicMock()
    manager.initialise_pandapower_network(parameters)
    manager._initialise.assert_called_once()
    assert_equal(manager._parameters.problem_settings.opf_optimizer, "TEST")

def tes_initialise():
    manager = PandaPowerManager()
    manager.load_mat_file_to_pandapower = MagicMock()
    manager.add_controllers_to_network = MagicMock()
    manager.add_output_writer_to_network = MagicMock()
    manager.define_simulation_settings = MagicMock()
    manager.load_mat_file_to_pandapower.assert_called_once()
    manager.add_controllers_to_network.assert_called_once()
    manager.add_output_writer_to_network.assert_called_once()
    manager.define_simulation_settings.assert_called_once()    

def _load_test_parameter_case_9():
    parameters = Parameters()
    parameters.pandapower_mpc_settings.mat_file_path = get_path_case9_mat()
    parameters.pandapower_mpc_settings.system_frequency = 50.0
    parameters.pandapower_mpc_settings.initialised =  True
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
        indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
        columns=['load1_p']), active_columns_names=['load1_p'])]
    parameters.output_settings.initialised = True
    parameters.output_settings.output_variables = [OutputVariable(name_dataset='res_bus', \
        name_variable='p_mw', variable_indexes=[]), OutputVariable(name_dataset='res_line', \
        name_variable='i_ka', variable_indexes=[])]
    parameters.output_settings.directory = set_pandapower_test_output_directory()
    parameters.output_settings.format = ".xlsx"
    parameters.opf_time_settings.initialised = True
    parameters.opf_time_settings.date_time_settings = \
        date_range(start="00:00:00", end="02:00:00", freq="H")
    parameters.pandapower_optimisation_settings.initialised = True
    parameters.pandapower_optimisation_settings.continue_on_divergence = False
    parameters.pandapower_optimisation_settings.display_progress_bar = False
    parameters.pandapower_optimisation_settings.optimisation_software = 'pypower'
    parameters.problem_settings.initialised = True
    parameters.problem_settings.opf_type = "ac"
    return parameters

def test_run_timestep_opf_pandapower():
    manager = PandaPowerManager()
    parameters = _load_test_parameter_case_9()
    manager.initialise_pandapower_network(parameters)
    manager.run_timestep_opf_pandapower()
    assert manager.wrapper.network.OPF_converged == True
    assert isclose(3583.53647, manager.wrapper.network.res_cost, abs_tol=1e-4)

def test_update_network_controllers_case1():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
            columns=['load1_p']), active_columns_names=['load1_p']),
        PandaPowerProfileData(element_type="gen", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], \
            columns=['gen1_p']), active_columns_names=['gen1_p'])]
    manager._parameters = parameters
    manager.add_controllers_to_network()
    pp_profiles = PandaPowerProfilesData()
    pp_profiles.initialised = True
    pp_profiles.data = [\
        PandaPowerProfileData(element_type="gen", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[220], [190], [5]], \
            columns=['gen1_p']), active_columns_names=['gen1_p'])]
    manager.update_network_controllers(pp_profiles)
    RESULT = manager.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[220], [190], [5]], columns=['gen1_p']).equals(RESULT)

def test_update_network_controllers_case2():
    manager = PandaPowerManager()
    parameters = Parameters()
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
            columns=['load1_p']), active_columns_names=['load1_p']),
        PandaPowerProfileData(element_type="gen", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], \
            columns=['gen1_p']), active_columns_names=['gen1_p'])]
    manager._parameters = parameters
    manager.add_controllers_to_network()
    pp_profiles = PandaPowerProfilesData()
    pp_profiles.initialised = False
    manager.update_network_controllers(pp_profiles)
    RESULT = manager.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[240.44092015], [205.50525905], [18.7321705]], \
        columns=['gen1_p']).equals(RESULT)

def test_is_feasible():
    manager = PandaPowerManager()
    manager.wrapper.is_feasible = MagicMock()
    manager.wrapper.is_feasible.return_value = True
    assert manager.is_feasible() == True
