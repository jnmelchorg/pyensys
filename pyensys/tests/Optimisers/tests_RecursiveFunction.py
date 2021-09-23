from numpy.lib.npyio import load
from pyensys.Optimisers.RecursiveFunction import RecursiveFunction, DataMultipliers, DataMultiplier
from pyensys.readers.ReaderDataClasses import Parameters, PandaPowerProfileData, OutputVariable
from pyensys.tests.tests_data_paths import get_path_case9_mat, set_pandapower_test_output_directory
from pandas import DataFrame, date_range
from math import isclose
from typing import List
from copy import copy

def load_test_case() -> Parameters:
    parameters = Parameters()
    parameters.pandapower_mpc_settings.mat_file_path = get_path_case9_mat()
    parameters.pandapower_mpc_settings.system_frequency = 50.0
    parameters.pandapower_mpc_settings.initialised =  True
    parameters.pandapower_profiles_data.initialised = True
    parameters.pandapower_profiles_data.data = [\
        PandaPowerProfileData(element_type="load", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[67.28095505], [9.65466896], [11.70181664]], \
            columns=['load1_p']), active_columns_names=['load1_p']),
        PandaPowerProfileData(element_type="gen", variable_name="p_mw", \
            indexes=[0], data=DataFrame(data=[[240], [200], [18]], \
            columns=['gen1_p']), active_columns_names=['gen1_p'])]
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
    parameters.problem_settings.opf_optimizer = "pandapower"
    parameters.problem_settings.intertemporal = True
    return parameters

def load_data_multipliers() -> List[DataMultipliers]:
    list_multipliers = []
    data_multipliers = DataMultipliers()
    data_multipliers.initialised = True
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.1, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.2, element_type="gen", variable_name="p_mw")]
    list_multipliers.append(copy(data_multipliers))
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.3, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.4, element_type="gen", variable_name="p_mw")]
    list_multipliers.append(data_multipliers)
    return list_multipliers

def test_initialise():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    assert len(RF.pp_opf.wrapper.network['bus'].index) == 9
    assert len(RF.pp_opf.wrapper.network['controller'].iat[0, 0].data_source.df.index) == 3
    assert len(RF.pp_opf.wrapper.network['output_writer'].iat[0, 0].log_variables) == 2
    assert len(RF.pp_opf.simulation_settings.time_steps) == 3
    assert RF.pp_opf.simulation_settings.opf_type == "ac"
    assert RF.pp_opf.simulation_settings.optimisation_software == "pypower"

def test_operational_check():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF.operational_check()
    assert RF.pp_opf.wrapper.network.OPF_converged == True
    assert isclose(3583.53647, RF.pp_opf.wrapper.network.res_cost, abs_tol=1e-4)

def test_get_data_multipliers_current_node():
    RF = RecursiveFunction()
    data_multipliers = DataMultipliers()
    data_multipliers.initialised = True
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.1, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.2, element_type="load", variable_name="p_mw")]
    RF.graph_nodes_multipliers.append(data_multipliers)
    data_multipliers.multipliers = [\
        DataMultiplier(data_multiplier=0.3, element_type="load", variable_name="p_mw"), \
        DataMultiplier(data_multiplier=0.4, element_type="load", variable_name="p_mw")]
    RF.graph_nodes_multipliers.append(data_multipliers)
    assert RF.get_data_multipliers_current_node(1).multipliers[0].data_multiplier == 0.3

def test_get_profile_position_to_update():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    list_multipliers = load_data_multipliers()
    PROFILE_POSITION = RF.get_profile_position_to_update(list_multipliers[1].multipliers[1])
    assert PROFILE_POSITION == 1



def test_update_pandapower_controllers():
    RF = RecursiveFunction()
    parameters = load_test_case()
    RF.initialise(parameters)
    RF.graph_nodes_multipliers = load_data_multipliers()
    RF.update_pandapower_controllers(1)
    RESULT = RF.pp_opf.wrapper.network['controller'].iat[1, 0].data_source.df
    assert DataFrame(data=[[96], [80], [7.2]], columns=['gen1_p']).equals(RESULT)

