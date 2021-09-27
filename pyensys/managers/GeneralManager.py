from pyensys.readers.ReaderManager import read_parameters
from pyensys.managers.DistributionSystemManager import identify_and_solve

def main_access_function(file_path: str):
    parameters = read_parameters(file_path)
    if parameters.problem_settings.system == "distribution":
        identify_and_solve(parameters)

        
        
