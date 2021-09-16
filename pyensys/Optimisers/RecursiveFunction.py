from pyensys.wrappers.pandapower import PandaPowerWrapper
from pyensys.readers.ReaderDataClasses import Parameters

class RecursiveFunction:

    def feasibility_check() -> bool:
        pass

def initialise_pandapower_network(problem_parameters: Parameters) -> PandaPowerWrapper:
    wrapper = PandaPowerWrapper()
    mpc_settings = problem_parameters.pandapower_mpc_settings
    wrapper.load_mat_file_to_pandapower(filename_with_extension=mpc_settings.mat_file_path, \
        frequency_hz=mpc_settings.system_frequency)
    profiles = problem_parameters.profiles_data.data
    for profile in profiles:
        pass

